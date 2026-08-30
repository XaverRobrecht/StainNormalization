import torch


@torch.no_grad()
def _gabor_kernel(theta, sigma=1.0, Lambda=10.0, psi=0.0, gamma=1.2, ksize=7):
    """Generate Gabor filter kernel.

    Parameters
    ----------
    theta : float
        Orientation.
    sigma : float, optional
        Standard deviation, by default 1.0.
    Lambda : float, optional
        Wavelength, by default 10.0.
    psi : float, optional
        Phase offset, by default 0.0.
    gamma : float, optional
        Aspect ratio, by default 1.2.
    ksize : int, optional
        Kernel size, by default 7.

    Returns
    -------
    torch.Tensor
        Gabor kernel.
    """
    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    radius = ksize // 2
    coordinates = torch.arange(-radius, radius + 1)

    y, x = torch.meshgrid(
        coordinates,
        coordinates,
        indexing="ij",
    )

    x_theta = x * torch.cos(theta) + y * torch.sin(theta)
    y_theta = -x * torch.sin(theta) + y * torch.cos(theta)

    gb = torch.exp(-0.5 * (x_theta**2 / sigma_x**2 + y_theta**2 / sigma_y**2)) * torch.cos(
        2 * torch.pi / Lambda * x_theta + psi
    )
    return gb


class ScanProcessor(torch.nn.Module):
    """Stain decomposition and normalization processor for histopathology images.
    
    This module implements stain color normalization based on stain component separation
    and reassembly. It decomposes RGB images into hematoxylin (nuclei) and eosin (stroma)
    stain components, then normalizes them to match a reference image's staining profile.
    
    The processor must be fitted on a reference image before use, which establishes
    the target staining characteristics. Forward pass then applies normalization
    to input images by decomposing, rescaling, and recomposing stain components.
    """
    def __init__(self, reference_image=None, n_iterations=10, device="cpu"):
        """Initialize ScanProcessor.
        
        Parameters
        ----------
        reference_image : torch.Tensor, optional
            Reference RGB image tensor for fitting. If provided, fit() is called automatically.
            Should be shape (3,H,W) or (B,3,H,W), by default None.
        n_iterations : int, optional
            Number of iterations for stain vector refinement, by default 10.
        device : str, optional
            Device to place module on ('cpu' or 'cuda'), by default "cpu".
        """
        super().__init__()
        self.iterations = n_iterations
        self.macenko_estimator = MacenkoEstimator()
        self.white_detector = WhiteDetector()
        self.nuclei_segmentor = NucleiSegmentor()
        self.stroma_segmentor = StromaSegmentor()
        self.to(device)

        self.h_rm = None
        self.w_ref = None
        self._is_fitted = False

        if reference_image is not None:
            self.fit(reference_image.to(device))

    def fit(self, reference_image):
        """Fit processor to reference image for normalization.
        
        Decomposes the reference image to extract its stain profile (hematoxylin and eosin
        stain vectors and intensities), which serves as the target for normalizing other images.
        
        Parameters
        ----------
        reference_image : torch.Tensor
            Reference RGB image tensor, shape (3,H,W) or (B,3,H,W).
            
        Returns
        -------
        ScanProcessor
            Self for method chaining.
        """
        if reference_image.ndim == 3:
            reference_image = reference_image.unsqueeze(0)

        h_est, west, _, _ = self.deconvolve(reference_image)
        b, c, other = h_est.shape
        h_rm = torch.quantile(
            h_est.reshape((b * c, other)),
            dim=1,
            q=torch.as_tensor([99.0 / 100.0], dtype=torch.float32, device=reference_image.device),
        ).T.reshape((b,c,1))

        with torch.no_grad():
            self.h_rm = torch.nn.Parameter(h_rm, requires_grad=False)
            self.w_ref = torch.nn.Parameter(west, requires_grad=False)

        self._is_fitted = True
        return self

    def forward(self, x):
        """Normalize input image to reference staining profile.
        
        Decomposes input image into stain components, rescales them to match the reference
        profile, and recomposes the normalized RGB image. Also returns intermediate
        segmentation masks for nuclei and stroma regions, plus individual stain intensities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input RGB image tensor, shape (3,H,W) or (B,3,H,W).
            
        Returns
        -------
        tuple
            Tuple containing:
            - normalized_image : torch.Tensor
                Stain-normalized RGB image, clipped to [0, 1].
            - nuclei_mask : torch.Tensor
                Binary segmentation mask for nuclei regions.
            - stroma_mask : torch.Tensor
                Binary segmentation mask for stroma regions.
            - hematoxylin : torch.Tensor
                Hematoxylin stain intensity map.
            - eosin : torch.Tensor
                Eosin stain intensity map.
                
        Raises
        ------
        RuntimeError
            If processor has not been fitted with a reference image.
        """
        if not self._is_fitted or self.h_rm is None or self.w_ref is None:
            raise RuntimeError(
                "ScanProcessor must be fitted before calling forward(). "
                "Call fit(reference_image_tensor) first."
            )
        if x.ndim == 3:
            x = x.unsqueeze(0)
        b, ch, hi, wd = x.shape

        h_scaled, _, nuclei_mask, stroma_mask = self.deconvolve(x)
        b, c, other = h_scaled.shape
        h_rm = torch.quantile(
            h_scaled.reshape((b * c, other)),
            torch.as_tensor([99.0 / 100.0], dtype=torch.float32, device=x.device),
            dim=1,
        ).T.reshape((b, 2, 1))

        h = h_scaled / h_rm * self.h_rm
        v = self.w_ref @ h

        hem = h[:, [0], :].reshape((b, 1, hi, wd))
        eos = h[:, [1], :].reshape((b, 1, hi, wd))

        return (
            torch.clip((10 ** (-v)).reshape((b, ch, hi, wd)), 0.0, 1.0),
            nuclei_mask,
            stroma_mask,
            hem,
            eos,
        )

    def deconvolve(self, x):
        """Deconvolve image into stain components.

        Parameters
        ----------
        x : torch.Tensor
            Input image (B,3,H,W).

        Returns
        -------
        tuple
            (stain_densities, stain_vectors, nuclei_mask, stroma_mask)
        """
        b, c, h, w = x.shape

        eps = torch.as_tensor(1e-9, dtype=torch.float32, device=x.device)
        image_od = -torch.log10(torch.maximum(x, eps))
        nonwhite_mask = ~self.white_detector(x)

        w_initial = self.macenko_estimator(image_od, nonwhite_mask)

        # stain separation
        w_est = w_initial
        image_od = torch.where(nonwhite_mask, image_od,0)
        for _ in range(self.iterations):

            h_est = torch.linalg.pinv(w_est) @ image_od.reshape((b,3, -1))

            # stain separation
            ## get masks
            hem_intensities = h_est[:, [0], :].reshape(b, 1, h, w)
            eos_intensities = h_est[:, [1], :].reshape(b, 1, h, w)
            stroma_mask = self.stroma_segmentor(
                torch.maximum(eos_intensities - hem_intensities, 0 * eps),
                nonwhite_mask[:, [0], :, :],
            )
            nuclei_mask = self.nuclei_segmentor(
                torch.maximum(hem_intensities - eos_intensities, 0 * eps),
                nonwhite_mask[:, [0], :, :],
            )

            # keep masks mutually exclusive
            # exclusive = torch.logical_xor(stroma_mask, nuclei_mask)
            stroma_mask = stroma_mask & nonwhite_mask[:, [0], :, :]  # & exclusive
            nuclei_mask = nuclei_mask & nonwhite_mask[:, [0], :, :] & ~stroma_mask

            ## estimate stain vectors via median
            masked_od = image_od.masked_fill(~nuclei_mask, float("nan")).reshape((b,c,-1))
            v_nuclei = torch.nanmedian(masked_od,dim=-1).values
            masked_od = image_od.masked_fill(~stroma_mask, float("nan")).reshape((b,c,-1))
            v_stroma = torch.nanmedian(masked_od,dim=-1).values
            w_new = torch.stack([v_nuclei, v_stroma], dim=2)
            w_est = w_new / torch.linalg.vector_norm(w_new, dim=1, keepdim=True)
        hem = h_est[:, [0], :] * w_est[:, :, [0]]
        eos = h_est[:, [1], :] * w_est[:, :, [1]]
        # H scaling
        h_scaled = torch.stack(
            [
                torch.median(hem / w_est[:, :, [0]], dim=1).values,
                torch.median(eos / w_est[:, :, [1]], dim=1).values,
            ],
            dim=1,
        )

        return (
            h_scaled,
            w_est,
            nuclei_mask,
            stroma_mask,
        )


class WienerFilter(torch.nn.Module):
    """Single-channel Wiener filter for noise reduction.
    
    Implements a local adaptive Wiener filter equivalent to scipy.signal.wiener.
    The filter reduces noise while preserving edges by using local mean and variance
    estimates. Particularly useful for preprocessing stain intensity maps before
    segmentation.
    """

    def __init__(self, kernel_size):
        """Initialize Wiener filter.

        Parameters
        ----------
        kernel_size : int or tuple
            Filter kernel size.
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = tuple(kernel_size)
        self.mean_filter = torch.nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel_size,
            padding="same",
            bias=False,
        )

        with torch.no_grad():
            weight = torch.ones(1, 1, *self.kernel_size, dtype=torch.float32)
            weight /= float(self.kernel_size[0] * self.kernel_size[1])
            self.mean_filter.weight.copy_(weight)
        self.mean_filter.weight.requires_grad_(False)

    def forward(self, images):
        """Apply Wiener filter.

        Parameters
        ----------
        images : torch.Tensor
            Input image(s).

        Returns
        -------
        torch.Tensor
            Filtered image(s).
        """

        if images.dim() == 2:
            images = images[None, None, :, :]
        elif images.dim() == 3:
            images = images[None, :, :, :]

        b, c, h, w = images.shape
        images = images.reshape((b * c, 1, h, w))

        local_mean = self.mean_filter(images)
        local_var = self.mean_filter(images**2) - local_mean**2
        noise = local_var.mean(dim=(-1, -2), keepdim=True)

        eps = torch.as_tensor(1e-9, dtype=torch.float32, device=images.device)
        normalised_var = torch.clamp(local_var, min=eps)
        correction = 1.0 - noise / normalised_var
        result = (images - local_mean) * correction + local_mean

        output = torch.where(local_var < noise, local_mean, result)
        return output.reshape((b, c, h, w))


class WhiteDetector(torch.nn.Module):
    """Detect background (white) regions in histopathology images.
    
    Uses Gabor filter responses across multiple orientations to identify white/background
    regions that should be excluded from stain analysis. Regions with high combined
    Gabor response are classified as background.
    """
    
    def __init__(self):
        """Initialize white region detector using Gabor filters.
        
        Sets up 8 Gabor filters at different orientations and luminance weights
        for RGB to grayscale conversion.
        """
        super().__init__()
        weight_tensor = torch.as_tensor(
            [299.0 / 1000, 587.0 / 1000, 114.0 / 1000], dtype=torch.float32
        ).reshape((1, 3, 1, 1))
        self.luminance_weights = torch.nn.parameter.Parameter(weight_tensor, requires_grad=False)
        filter_kernels = [_gabor_kernel(theta) for theta in torch.arange(8) / 8.0 * torch.pi]
        self.kernel = torch.nn.Parameter(
            torch.stack(filter_kernels, dim=0).unsqueeze(1), requires_grad=False
        )

    def forward(self, x):
        """Detect white regions in image.

        Parameters
        ----------
        x : torch.Tensor
            RGB image (B,3,H,W).

        Returns
        -------
        torch.Tensor
            Binary white mask.
        """

        b, c, h, w = x.shape
        lum = torch.sum(x * self.luminance_weights, dim=1, keepdim=True)
        filtered = torch.nn.functional.conv2d(lum, self.kernel, padding="same")
        y = filtered.sum(dim=1, keepdim=True)
        white_mask = y > (y.reshape((b, 1, -1)).max(dim=2).values * 0.9).reshape((b, 1, 1, 1))
        return torch.cat(3*[white_mask.reshape(b, 1, h, w)],dim=1)


class MacenkoEstimator(torch.nn.Module):
    """Estimate stain vectors using the Macenko color deconvolution method.
    
    Implements the Macenko stain normalization technique for histopathology images.
    Estimates hematoxylin and eosin stain vectors from optical density values using
    singular value decomposition and angular quantile selection.
    """
    
    def __init__(self):
        """Initialize Macenko stain estimator."""
        super().__init__()

    def forward(self, image_od, nonwhite_mask):
        """Estimate stain vectors.

        Parameters
        ----------
        image_od : torch.Tensor
            Optical density (batch) .
        nonwhite_mask : torch.Tensor
            Non-white masks (batch).

        Returns
        -------
        torch.Tensor
            Normalized stain vectors.
        """
        eps = torch.as_tensor(1e-9, dtype=torch.float32, device=image_od.device)
        b,c,h,w = image_od.shape
        image_nonwhite_od = (image_od*nonwhite_mask).reshape((b,c,h*w))
        u, _, _ = torch.linalg.svd(image_nonwhite_od, full_matrices=False)
        v1, v2 = u[:,:, 0].reshape((b,c, 1)), u[:,:, 1].reshape((b,c, 1))

        v1 = torch.abs(v1)
        v1 = v1 / torch.linalg.vector_norm(v1, axis=1,keepdim=True)
        v2 = v2 / torch.linalg.vector_norm(v2, axis=1,keepdim=True)

        projection1 = torch.sum(image_nonwhite_od * v1, axis=1)
        projection2 = torch.sum(image_nonwhite_od * v2, axis=1)

        angles = torch.arctan2(projection2, projection1)
        angles_for_quantile = torch.where(nonwhite_mask[:,0].reshape(b,h*w), angles, torch.full_like(angles, torch.nan))
        q = torch.as_tensor([1.0 / 100.0, 99.0 / 100.0], device=angles.device)
        a1, a2 = torch.nanquantile(angles_for_quantile, q, dim=1)

        a1 = a1.reshape((b,1,1))
        a2 = a2.reshape((b,1,1))

        od1 = torch.cos(a1) * v1 + torch.sin(a1) * v2
        od2 = torch.cos(a2) * v1 + torch.sin(a2) * v2

        #Hematoxylin is more blue -> use that to ensure consistent ordering of stain vectors
        mask = (od1[:, 2] - od1[:, 0] <= od2[:, 2] - od2[:, 0]).reshape((b, 1, 1))
        vec1 = torch.where(mask, od1, od2)
        vec2 = torch.where(mask, od2, od1)
        w_est = torch.stack([vec1, vec2], axis=2)
        w_est = w_est / torch.maximum(torch.linalg.norm(w_est, dim=1, keepdim=True), eps)
        return w_est.squeeze(-1) 


class StromaSegmentor(torch.nn.Module):
    """Segment stroma (connective tissue) regions using k-means clustering.
    
    Performs batch k-means clustering on eosin stain intensities to separate
    stroma from nuclei regions. Stroma regions typically have higher eosin
    intensity compared to nuclei stained with hematoxylin.
    """
    
    def __init__(self):
        """Initialize stroma segmentor."""
        super().__init__()

    def forward(self, eos_intensities, mask, n_iters=25):
        """Batch segmentation of stroma regions via k-means on eosin intensities.

        Parameters
        ----------
        eos_intensities : torch.Tensor
            Eosin channel values, shape (B, 1, H, W).
        mask : torch.Tensor
            Binary foreground mask, shape (B, 1, H, W).
        n_iters : int
            Number of K means iterations.


        Returns
        -------
        torch.Tensor
            Stroma mask, shape (B, 1, H, W).
        """
        b, c, h, w = eos_intensities.shape
        eos = eos_intensities.reshape(b, -1)                       # (B, N)
        valid = (mask.reshape(b, -1) & (eos > 0))                  # (B, N) bool
        weight = valid.to(torch.float32)

        # --- init cluster centers: 1st/99th percentile of VALID eos values ---
        eos_for_quantile = torch.where(valid, eos, torch.full_like(eos, torch.nan))
        q = torch.as_tensor([1.0 / 100.0, 99.0 / 100.0], device=eos.device)
        clusters = torch.nanquantile(eos_for_quantile, q, dim=1).T  # (B, 2)  (nanquantile puts q first -> transpose)

        for _ in range(n_iters):
            distances = (clusters.unsqueeze(2) - eos.unsqueeze(1)) ** 2   # (B, 2, N)
            idx = torch.argmin(distances, dim=1)                           # (B, N), arbitrary where invalid — fine, weighted out below

            is0 = (idx == 0).to(torch.float32) * weight
            is1 = (idx == 1).to(torch.float32) * weight
            counts = torch.stack([is0.sum(dim=1), is1.sum(dim=1)], dim=1).clamp(min=1.0)  # (B, 2)

            sums = torch.stack([(eos * is0).sum(dim=1), (eos * is1).sum(dim=1)], dim=1)   # (B, 2)
            clusters = sums / counts

        distances = (clusters.unsqueeze(2) - eos.unsqueeze(1)) ** 2
        idx = torch.argmin(distances, dim=1)                        # (B, N)

        bright_idx = torch.argmax(clusters, dim=1, keepdim=True)    # (B, 1)
        is_bright = (idx == bright_idx)                              # (B, N)

        return_mask = (is_bright & valid).reshape(b, c, h, w)
        return return_mask


class NucleiSegmentor(torch.nn.Module):
    """Segment cell nuclei regions using Otsu's thresholding method.
    
    Applies local adaptive noise filtering via Wiener filter followed by histogram-based
    Otsu thresholding to identify cell nuclei regions. Uses Bayesian class separation to
    determine optimal threshold by minimizing within-class variance.
    """
    
    def __init__(self,kernelsize=3):
        """Initialize nuclei segmentor with Wiener filter.
        
        Parameters
        ----------
        kernelsize : int, optional
            Kernel size for the Wiener filter, by default 3.
        """
        super().__init__()
        self.wiener = WienerFilter(kernelsize)

    def forward(self, gs_image_values, nonwhite_mask):
        """Segment nuclei regions.

        Parameters
        ----------
        gs_image_values : torch.Tensor
            Hematoxylin channel (B,1,H,W).
        nonwhite_mask : torch.Tensor
            Non-white foreground mask.

        Returns
        -------
        torch.Tensor
            Nuclei binary mask.
        """
        eps = torch.as_tensor(1e-9, dtype=torch.float32, device=gs_image_values.device)
        b, c, h, w = gs_image_values.shape
        input_batch = gs_image_values.reshape((b, -1))
        mins = input_batch.min(dim=1, keepdims=True).values
        maxs = input_batch.max(dim=1, keepdims=True).values
        input_batch = (input_batch - mins) / torch.maximum(maxs - mins, eps)
        gs_flat = self.wiener(input_batch.reshape(gs_image_values.shape)).reshape(( b,-1))
        mask_flat = nonwhite_mask.reshape(b, -1)          # (B, N)

        nbins = 128
        edges = torch.arange(nbins + 1, dtype=torch.float32, device=gs_flat.device) / nbins
        bin_idx = torch.bucketize(gs_flat, edges) - 1       # (B, N)
        in_range = (bin_idx >= 0) & (bin_idx < nbins)
        valid = (in_range & mask_flat).to(torch.float32)    # weight, not a selector
        bin_idx = bin_idx.clamp(0, nbins - 1)

        # Histogram via scatter_add
        counts = torch.zeros(b, nbins, dtype=torch.float32, device=gs_flat.device)
        counts.scatter_add_(1, bin_idx, valid)

        total = torch.maximum(counts.sum(dim=1, keepdim=True), torch.as_tensor(1.0, device=gs_flat.device))
        prob = counts / total                                # (B, nbins)

        edges = edges[1:]
        cumulative_p = torch.cumsum(prob, dim=1)
        cumulative_xp = torch.cumsum(prob * edges, dim=1)
        cumulative_x2p = torch.cumsum(prob * edges**2, dim=1)

        p0 = cumulative_p[:, :-1]
        p1 = cumulative_p[:, -1:] - p0
        xp0 = cumulative_xp[:, :-1]
        x2p0 = cumulative_x2p[:, :-1]
        xp1 = cumulative_xp[:, -1:] - xp0
        x2p1 = cumulative_x2p[:, -1:] - x2p0

        safe_p0 = torch.maximum(p0, eps)
        safe_p1 = torch.maximum(p1, eps)

        Ex_0, Ex2_0 = xp0 / safe_p0, x2p0 / safe_p0
        Ex_1, Ex2_1 = xp1 / safe_p1, x2p1 / safe_p1

        var0 = torch.clamp(Ex2_0 - Ex_0**2, min=0.0)
        var1 = torch.clamp(Ex2_1 - Ex_1**2, min=0.0)

        energy = -(
            p0**2 * var0 * torch.log(torch.maximum(var0, eps))
            + p1**2 * var1 * torch.log(torch.maximum(var1, eps))
        )
        valid_bin = (p0 > eps) & (p1 > eps)
        energy = torch.where(valid_bin, energy, torch.full_like(energy, torch.inf))

        i_max = torch.argmin(energy, dim=1)                  # (B,)
        th = torch.gather(edges[:-1].expand(b, -1), 1, i_max.unsqueeze(1))  # (B, 1)
        th = th.reshape(b, 1, 1, 1)

        return (gs_flat.reshape((b,c,h,w)) > th) & nonwhite_mask