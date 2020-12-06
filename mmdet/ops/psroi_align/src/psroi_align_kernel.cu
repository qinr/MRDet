#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__device__ scalar_t bilinear_interpolate(const scalar_t *bottom_data,
                                         const int height, const int width,
                                         scalar_t y, scalar_t x) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;
  // do bilinear interpolation
  scalar_t lt = bottom_data[y_low * width + x_low];
  scalar_t rt = bottom_data[y_low * width + x_high];
  scalar_t lb = bottom_data[y_high * width + x_low];
  scalar_t rb = bottom_data[y_high * width + x_high];
  scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);

  return val;
}

// 平均池化
template <typename scalar_t>
__global__ void PSROIAlignForward(
    const int nthreads, const scalar_t *bottom_data, const scalar_t *rois,
    const scalar_t spatial_scale, const int sample_num,
    const int channels, const int height,
    const int width, const int pooled_h, const int pooled_w,
    const int group_size, const int out_chn, scalar_t *top_data,
    int *mapping_channel) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, ctop, ph, pw) is an element in the pooled output
    int pw = index % pooled_w;
    int ph = (index / pooled_w) % pooled_h;
    int ctop = (index / pooled_w / pooled_h) % out_chn;
    int n = index / pooled_w / pooled_h / out_chn;

    const scalar_t *offset_bottom_rois = rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    // calculate the roi region on feature maps
    scalar_t roi_x1 = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_y1 = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_x2 = (offset_bottom_rois[3] + 1.) * spatial_scale;
    scalar_t roi_y2 = (offset_bottom_rois[4] + 1.) * spatial_scale;

    // force malformed rois to be 1x1
    scalar_t roi_w = max(roi_x2 - roi_x1, scalar_t(1.));
    scalar_t roi_h = max(roi_y2 - roi_y1, scalar_t(1.));

    scalar_t bin_size_w =
        static_cast<scalar_t>(roi_w) / static_cast<scalar_t>(pooled_w);
    scalar_t bin_size_h =
        static_cast<scalar_t>(roi_h) / static_cast<scalar_t>(pooled_h);

    // the corresponding bin region
    scalar_t bin_x1 = static_cast<scalar_t>(pw) * bin_size_w + roi_x1;
    scalar_t bin_y1 = static_cast<scalar_t>(ph) * bin_size_h + roi_y1;
    scalar_t bin_x2 = static_cast<scalar_t>(pw + 1) * bin_size_w + roi_x1;
    scalar_t bin_y2 = static_cast<scalar_t>(ph + 1) * bin_size_h + roi_y1;

    // add roi offsets and clip to input boundaries
    bin_x1 = min(max(bin_x1, 0.0), static_cast<scalar_t>(width));
    bin_y1 = min(max(bin_y1, 0.0), static_cast<scalar_t>(height));
    bin_x2 = min(max(bin_x2, 0.0), static_cast<scalar_t>(width));
    bin_y2 = min(max(bin_y2, 0.0), static_cast<scalar_t>(height));
    bool is_empty = (bin_y2 <= bin_y1) || (bin_x2 <= bin_x1);

    // the corresponding input channel
    int gw = pw;
    int gh = ph;
    int c = (ctop * group_size + gh) * group_size + gw;

    const scalar_t *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;
    int sample_num_h = (sample_num > 0)
                           ? sample_num
                           : ceil(roi_h/ pooled_h);  // e.g., = 2
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_w / pooled_w);

    scalar_t out_sum = 0.0;
    for (int iy = 0; iy < sample_num_h; iy++) {
      const scalar_t y = bin_y1 + (scalar_t)(iy + scalar_t(.5f)) * bin_size_h /
                             (scalar_t)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        const scalar_t x = bin_x1 + (scalar_t)(ix + scalar_t(.5f)) * bin_size_w /
                               (scalar_t)(sample_num_w);
        scalar_t val = bilinear_interpolate<scalar_t>(offset_bottom_data,
                                                      height, width, y, x);
        out_sum += val;
      }
    }
    top_data[index] = is_empty ? (scalar_t)(0.) : out_sum / static_cast<scalar_t>(sample_num_h * sample_num_w);
    mapping_channel[index] = c;
  }
}

int PSROIAlignForwardLauncher(const at::Tensor features, const at::Tensor rois,
                             const float spatial_scale, const int sample_num,
                             const int channels,
                             const int height, const int width,
                             const int num_rois, const int pooled_h,
                             const int pooled_w, const int group_size,
                             const int out_chn, at::Tensor output,
                             at::Tensor mapping_channel) {
  const int output_size = num_rois * out_chn * pooled_h * pooled_w;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      features.type(), "PSROIAlignLaucherForward", ([&] {
        const scalar_t *bottom_data = features.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();
        int *mapping_channel_data = mapping_channel.data<int>();

        PSROIAlignForward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, bottom_data, rois_data, scalar_t(spatial_scale),
                sample_num, channels, height, width, pooled_h, pooled_w, group_size,
                out_chn, top_data, mapping_channel_data);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  return 1;
}

template <typename scalar_t>
__device__ void bilinear_interpolate_gradient(const int height, const int width,
                                              scalar_t y, scalar_t x,
                                              scalar_t &w1, scalar_t &w2,
                                              scalar_t &w3, scalar_t &w4,
                                              int &x_low, int &x_high,
                                              int &y_low, int &y_high) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (scalar_t)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (scalar_t)x_low;
  } else {
    x_high = x_low + 1;
  }

  scalar_t ly = y - y_low;
  scalar_t lx = x - x_low;
  scalar_t hy = 1. - ly;
  scalar_t hx = 1. - lx;

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename scalar_t>
__global__ void PSROIAlignBackward(
    const int nthreads, const scalar_t *top_diff, const scalar_t *rois,
    const int *mapping_channel, const scalar_t spatial_scale, const int sample_num,
    const int channels, const int height, const int width, const int pooled_h,
    const int pooled_w, const int out_chn, scalar_t *bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_w;
    int ph = (index / pooled_w) % pooled_h;
    int n = index / pooled_w / pooled_h / out_chn;

    const scalar_t *offset_bottom_rois = rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    scalar_t roi_x1 = offset_bottom_rois[1] * spatial_scale;
    scalar_t roi_y1 = offset_bottom_rois[2] * spatial_scale;
    scalar_t roi_x2 = (offset_bottom_rois[3] + 1.) * spatial_scale;
    scalar_t roi_y2 = (offset_bottom_rois[4] + 1.) * spatial_scale;

    // Force too small ROIs to be 1x1
    scalar_t roi_w = max(roi_x2 - roi_x1, (scalar_t)1.);
    scalar_t roi_h = max(roi_y2 - roi_y1, (scalar_t)1.);

    // Compute w and h at bottom
    scalar_t bin_size_w =
        static_cast<scalar_t>(roi_w) / static_cast<scalar_t>(pooled_w);
    scalar_t bin_size_h =
        static_cast<scalar_t>(roi_h) / static_cast<scalar_t>(pooled_h);

    scalar_t bin_x1 = static_cast<scalar_t>(pw) * bin_size_w + roi_x1;
    scalar_t bin_y1 = static_cast<scalar_t>(ph) * bin_size_h + roi_y1;
    scalar_t bin_x2 = static_cast<scalar_t>(pw + 1) * bin_size_w + roi_x1;
    scalar_t bin_y2 = static_cast<scalar_t>(ph + 1) * bin_size_h + roi_y1;

    // add roi offsets and clip to input boundaries
    bin_x1 = min(max(bin_x1, 0.0), static_cast<scalar_t>(width));
    bin_y1 = min(max(bin_y1, 0.0), static_cast<scalar_t>(height));
    bin_x2 = min(max(bin_x2, 0.0), static_cast<scalar_t>(width));
    bin_y2 = min(max(bin_y2, 0.0), static_cast<scalar_t>(height));
    bool is_empty = (bin_y2 <= bin_y1) || (bin_x2 <= bin_x1);

    // Compute c at bottom
    int c = mapping_channel[index];
    scalar_t *offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;
    int sample_num_h = (sample_num > 0)
                           ? sample_num
                           : ceil(roi_h / pooled_h);  // e.g., = 2
    int sample_num_w =
        (sample_num > 0) ? sample_num : ceil(roi_w / pooled_w);
    const scalar_t count = (scalar_t)(sample_num_h * sample_num_w);
    scalar_t diff_val = is_empty ? static_cast<scalar_t>(0.) : top_diff[index];

     for (int iy = 0; iy < sample_num_h; iy++) {
      const scalar_t y = bin_y1 + (scalar_t)(iy + .5f) * bin_size_h / (scalar_t)(sample_num_h);
      for (int ix = 0; ix < sample_num_w; ix++) {
        const scalar_t x = bin_x1 + (scalar_t)(ix + .5f) * bin_size_w / (scalar_t)(sample_num_w);
        scalar_t w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient<scalar_t>(
            height, width, y, x, w1, w2, w3, w4, x_low, x_high, y_low, y_high);
        scalar_t g1 = diff_val * w1 / count;
        scalar_t g2 = diff_val * w2 / count;
        scalar_t g3 = diff_val * w3 / count;
        scalar_t g4 = diff_val * w4 / count;
        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, g1);
          atomicAdd(offset_bottom_diff + y_low * width + x_high, g2);
          atomicAdd(offset_bottom_diff + y_high * width + x_low, g3);
          atomicAdd(offset_bottom_diff + y_high * width + x_high, g4);
        }
      }
    }
  }
}



int PSROIAlignBackwardLauncher(const at::Tensor top_grad, const at::Tensor rois,
                              const at::Tensor mapping_channel,
                              const float spatial_scale, const int sample_num,
                              const int channels,
                              const int height, const int width,
                              const int num_rois, const int pooled_h,
                              const int pooled_w, const int out_chn,
                              at::Tensor bottom_grad) {
  const int output_size = num_rois * out_chn * pooled_h * pooled_w;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      top_grad.type(), "PSROIAlignLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        const int *mapping_channel_data = mapping_channel.data<int>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();

        if (sizeof(scalar_t) == sizeof(double)) {
          fprintf(stderr, "double is not supported\n");
          exit(-1);
        }

        PSROIAlignBackward<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
                output_size, top_diff, rois_data, mapping_channel_data,
                scalar_t(spatial_scale), sample_num, channels, height, width, pooled_h,
                pooled_w, out_chn, bottom_diff);
      }));
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  return 1;
}
