import pyopencl as cl
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.transform import warp
import logging
from datetime import datetime
from collections import deque
import math


class MergeData:
    def __init__(self, target, target_tag, query, query_tag, distances, minimize=True):
        self.target = target
        self.target_tag = target_tag
        self.query = query
        self.query_tag = query_tag
        self.distances = distances
        self.minimize = minimize

    def tag(self):
        return self.target_tag | self.query_tag

    @staticmethod
    def popcount(x):
        return bin(x).count('1')

    def _min_indices(self):
        mat = np.nansum(self.distances, axis=3)
        return np.unravel_index(np.nanargmin(mat), mat.shape)

    def _max_indices(self):
        mat = np.nansum(self.distances, axis=3)
        return np.unravel_index(np.nanargmax(mat), mat.shape)

    def displacement(self):
        if self.minimize:
            rot_i, ty_i, tx_i = self._min_indices()
        else:
            rot_i, ty_i, tx_i = self._max_indices()
        return 2 * (rot_i - 5), ty_i - (self.distances.shape[1] - 1) / 2, tx_i - (self.distances.shape[2] - 1) / 2

    def min_distance(self):
        if self.minimize:
            rot_i, ty_i, tx_i = self._min_indices()
        else:
            rot_i, ty_i, tx_i = self._max_indices()
        return self.distances[rot_i, ty_i, tx_i]

    def size_delta(self):
        return (self.target.shape[0] - self.query.shape[0]) / 2, (self.target.shape[1] - self.query.shape[1]) / 2

    @staticmethod
    def _rotation_extension(height, width, rot):
        # shape extension forced by rotation
        rot_deg = abs(math.degrees(rot))
        inn1 = math.degrees(math.atan(height / width))
        inn2 = 90 - inn1
        ext_y = abs(math.sin(math.radians(rot_deg + inn1)) / math.sin(math.radians(inn1)))
        ext_x = abs(math.sin(math.radians(rot_deg + inn2)) / math.sin(math.radians(inn2)))
        return height * ext_y - height, width * ext_x - width

    def merged_image(self, weighted=False):
        if not weighted:
            target_weight = 0.5
            query_weight = 0.5
        else:
            target_weight = MergeData.popcount(self.target_tag)
            query_weight = MergeData.popcount(self.query_tag)
            total_weight = target_weight + query_weight
            target_weight /= total_weight
            query_weight /= total_weight

        min_deg, min_dy, min_dx = self.displacement()
        rot = math.radians(min_deg)

        ry, rx = MergeData._rotation_extension(self.query.shape[0], self.query.shape[1], rot)
        # both directions included

        target_center = self.target.shape[0] / 2, self.target.shape[1] / 2
        query_center = self.query.shape[0] / 2, self.query.shape[1] / 2

        ext_1 = max(-(target_center[0] - query_center[0] + min_dy - ry / 2), 0), \
                max(-(target_center[1] - query_center[1] + min_dx - rx / 2), 0)

        ext_2 = max((target_center[0] + query_center[0] + min_dy + ry / 2) - self.target.shape[0], 0), \
                max((target_center[1] + query_center[1] + min_dx + rx / 2) - self.target.shape[1], 0)

        output_shape = int(self.target.shape[0] + ext_1[0] + ext_2[0]), \
                       int(self.target.shape[1] + ext_1[1] + ext_2[1])

        # query transformation
        translation_to_origin = np.array([[1, 0, query_center[1]],
                                          [0, 1, query_center[0]],
                                          [0, 0, 1]])
        rotation_around_origin = np.array([[math.cos(rot), math.sin(rot), 0],
                                           [-math.sin(rot), math.cos(rot), 0],
                                           [0, 0, 1]])
        translation_to_output = np.array([[1, 0, -(ext_1[1] + target_center[1] + min_dx)],
                                          [0, 1, -(ext_1[0] + target_center[0] + min_dy)],
                                          [0, 0, 1]])
        transform = translation_to_origin.dot(rotation_around_origin).dot(translation_to_output)
        transformed_query = np.array(warp(self.query, transform, order=0, output_shape=output_shape, 
                                          mode='constant', preserve_range=True), dtype=np.float32)

        # target transformation
        translation_to_origin = np.array([[1, 0, target_center[1]],
                                          [0, 1, target_center[0]],
                                          [0, 0, 1]])
        translation_to_output = np.array([[1, 0, -(ext_1[1] + target_center[1])],
                                          [0, 1, -(ext_1[0] + target_center[0])],
                                          [0, 0, 1]])
        transform = translation_to_origin.dot(translation_to_output)
        transformed_target = np.array(warp(self.target, transform, order=0, output_shape=output_shape, 
                                           mode='constant', preserve_range=True), dtype=np.float32)

        # merging
        combined_mask = transformed_query[:, :, 3] * transformed_target[:, :, 3]
        combined_mask_3d = np.dstack([combined_mask for _ in range(4)])
        merged_image = transformed_target - target_weight * transformed_target * combined_mask_3d \
                     + transformed_query - query_weight * transformed_query * combined_mask_3d
        return merged_image


class Merger:
    def __init__(self, target_image, target_tag=1, dev_idx=-1):
        self.target_image = target_image
        self.target_tag = target_tag

        self.platform = cl.get_platforms()[0]
        self.device = self.platform.get_devices()[dev_idx]  # the last device in the list is most likely to be the GPU

        if not self.device.image_support:
            raise Exception('Device %s does not support image data type' % self.device)

        self.context = None
        self.program = None
        self.out_image = None
        self.query_image = None
        self.query_tag = -1

    def merge(self, query_image, query_tag=2):
        pass


class InplaceCcMerger:
    minimize = False

    def __init__(self, target_image, target_tag=1, dev_idx=-1):
        # dev_idx is ignored
        self.target_image = target_image
        self.target_tag = target_tag
        self.query_image = None
        self.query_tag = -1

    def merge(self, query_image, query_tag=2):
        self.query_image = query_image
        self.query_tag = query_tag

        target_mean = np.mean(self.target_image, axis=(0, 1))
        query_mean = np.mean(self.query_image, axis=(0, 1))

        mask1 = self.target_image[:, :, 3] * self.query_image[:, :, 3]
        mask3 = np.dstack([mask1, mask1, mask1, mask1])

        sum1 = np.sum((self.target_image - target_mean) * (self.query_image - query_mean) * mask3, axis=(0, 1))
        sum2 = np.sum((self.target_image - target_mean) ** 2, axis=(0, 1))
        sum3 = np.sum((self.query_image - query_mean) ** 2, axis=(0, 1))

        r = sum1 / (np.sqrt(sum2) * np.sqrt(sum3))

        out = np.empty((11, 1, 1, 3))
        out.fill(-1)
        out[5, 0, 0] = r[:3]
        return MergeData(self.target_image, self.target_tag, self.query_image,
                         self.query_tag, out, InplaceCcMerger.minimize)


class InplaceMseMerger:
    minimize = True

    def __init__(self, target_image, target_tag=1, dev_idx=-1):
        # dev_idx is ignored
        self.target_image = target_image
        self.target_tag = target_tag
        self.query_image = None
        self.query_tag = -1

    def merge(self, query_image, query_tag=2):
        self.query_image = query_image
        self.query_tag = query_tag

        out = np.empty((11, 1, 1, 3))
        out.fill(np.nan)
        out[5, 0, 0] = np.array([
            mean_squared_error(self.target_image[:, :, 0] * self.target_image[:, :, 3],
                               self.query_image[:, :, 0] * self.query_image[:, :, 3]),
            mean_squared_error(self.target_image[:, :, 1] * self.target_image[:, :, 3],
                               self.query_image[:, :, 1] * self.query_image[:, :, 3]),
            mean_squared_error(self.target_image[:, :, 2] * self.target_image[:, :, 3],
                               self.query_image[:, :, 2] * self.query_image[:, :, 3])
        ])

        return MergeData(self.target_image, self.target_tag, self.query_image,
                         self.query_tag, out, InplaceMseMerger.minimize)


class MseMerger(Merger):
    minimize = True
    kernel_path = 'resources/kernels/mean_squared_error.opencl'

    def merge(self, query_image, query_tag=2):
        self.query_image = query_image
        self.query_tag = query_tag

        self.context = cl.Context([self.device])

        target = cl.image_from_array(self.context, self.target_image, 4, mode='r')
        query = cl.image_from_array(self.context, self.query_image, 4, mode='r')
        out_dims = (query.shape[0] // 4) * 2 + 1, (query.shape[1] // 4) * 2 + 1
        dest = cl.Image(self.context, cl.mem_flags.WRITE_ONLY,
                        cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                        shape=(out_dims[0], out_dims[1], 11))

        self.program = cl.Program(self.context, open(MseMerger.kernel_path).read()).build()
        with cl.CommandQueue(self.context) as queue:
            self.program.mse(queue, dest.shape, None,
                             target, np.array(target.shape, dtype=np.int32),
                             query, np.array(query.shape, dtype=np.int32),
                             dest, np.array(dest.shape[:2], dtype=np.int32)).wait()
            out = np.empty((11, dest.shape[1], dest.shape[0], 4), dtype=np.float32)
            cl.enqueue_copy(queue, out, dest, origin=(0, 0), region=dest.shape)
        return MergeData(self.target_image, self.target_tag, self.query_image, 
                         self.query_tag, out[:, :, :, :3], MseMerger.minimize)


class CcMerger(Merger):
    minimize = False
    kernel_path = 'resources/kernels/correlation_coefficient.opencl'

    def merge(self, query_image, query_tag=2):
        self.query_image = query_image
        self.query_tag = query_tag

        self.context = cl.Context([self.device])

        target = cl.image_from_array(self.context, self.target_image, 4, mode='r')
        query = cl.image_from_array(self.context, self.query_image, 4, mode='r')
        out_dims = (query.shape[0] // 4) * 2 + 1, (query.shape[1] // 4) * 2 + 1
        img1avg = cl.Image(self.context, cl.mem_flags.READ_WRITE,
                           cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                           shape=(out_dims[0], out_dims[1], 11))
        img2avg = cl.Image(self.context, cl.mem_flags.READ_WRITE,
                           cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                           shape=(out_dims[0], out_dims[1], 11))
        corrco = cl.Image(self.context, cl.mem_flags.WRITE_ONLY,
                          cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                          shape=(out_dims[0], out_dims[1], 11))

        self.program = cl.Program(self.context, open(CcMerger.kernel_path).read()).build()
        with cl.CommandQueue(self.context) as queue:
            self.program.avg(queue, img1avg.shape, None,
                             target, np.array(target.shape, dtype=np.int32),
                             query, np.array(query.shape, dtype=np.int32),
                             img1avg, img2avg, np.array(img1avg.shape[:2], dtype=np.int32)).wait()
            self.program.corr(queue, img1avg.shape, None,
                              target, np.array(target.shape, dtype=np.int32),
                              query, np.array(query.shape, dtype=np.int32),
                              img1avg, img2avg, corrco, np.array(img1avg.shape[:2], dtype=np.int32)).wait()
            out = np.empty((11, img2avg.shape[1], img2avg.shape[0], 4), dtype=np.float32)
            cl.enqueue_copy(queue, out, corrco, origin=(0, 0), region=img2avg.shape)
        return MergeData(self.target_image, self.target_tag, self.query_image, self.query_tag, out[:, :, :, :3], CcMerger.minimize)


class CcMergerFixed(CcMerger):
    """
    This is a proof of concept implementation to prove that you can cache the fixed part of
    displacement computation in the first method 'avg' and pass it to the next. However, this 
    method is actually slower, probably due to copying and accessing an additional image.

    Normal CcMerger: 39.7 ms ± 1.18 ms per loop (mean ± std. dev. of 100 runs, 10 loops each)
    Fixed CcMerger: 41.9 ms ± 884 µs per loop (mean ± std. dev. of 100 runs, 10 loops each)
    """
    minimize = False
    kernel_path = 'resources/kernels/correlation_coefficient_fixed.opencl'

    def merge(self, query_image, query_tag=2):
        self.query_image = query_image
        self.query_tag = query_tag

        self.context = cl.Context([self.device])

        target = cl.image_from_array(self.context, self.target_image, 4, mode='r')
        query = cl.image_from_array(self.context, self.query_image, 4, mode='r')
        out_dims = (query.shape[0] // 4) * 2 + 1, (query.shape[1] // 4) * 2 + 1
        img1avg = cl.Image(self.context, cl.mem_flags.READ_WRITE,
                           cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                           shape=(out_dims[0], out_dims[1], 11))
        img2avg = cl.Image(self.context, cl.mem_flags.READ_WRITE,
                           cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                           shape=(out_dims[0], out_dims[1], 11))
        fixed = cl.Image(self.context, cl.mem_flags.READ_WRITE,
                         cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                         shape=(out_dims[0], out_dims[1], 11))
        corrco = cl.Image(self.context, cl.mem_flags.WRITE_ONLY,
                          cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                          shape=(out_dims[0], out_dims[1], 11))

        self.program = cl.Program(self.context, open(CcMergerFixed.kernel_path).read()).build()
        with cl.CommandQueue(self.context) as queue:
            self.program.avg(queue, img1avg.shape, None,
                             target, np.array(target.shape, dtype=np.int32),
                             query, np.array(query.shape, dtype=np.int32),
                             img1avg, img2avg, fixed, 
                             np.array(img1avg.shape[:2], dtype=np.int32)).wait()
            self.program.corr(queue, img1avg.shape, None,
                              target, np.array(target.shape, dtype=np.int32),
                              query, np.array(query.shape, dtype=np.int32),
                              img1avg, img2avg, fixed, corrco, 
                              np.array(img1avg.shape[:2], dtype=np.int32)).wait()
            out = np.empty((11, img2avg.shape[1], img2avg.shape[0], 4), dtype=np.float32)
            cl.enqueue_copy(queue, out, corrco, origin=(0, 0), region=img2avg.shape)
        return MergeData(self.target_image, self.target_tag, self.query_image, self.query_tag, out[:, :, :, :3], CcMergerFixed.minimize)


class MergeRunner:
    def __init__(self, image_list, weighted=True):
        self.logger = logging.getLogger(MergeRunner.__name__)
        self.images = {1 << i: image for i, image in enumerate(image_list)}
        self.candidate_nodes = dict()
        self.min_nodes = []
        self.weighted = weighted
        self.best_mergers = []

    @staticmethod
    def _round_robin_even(d, n):
        permutations = []
        for i in range(n - 1):
            permutations += [[d[j], d[-j - 1]] for j in range(n // 2)]
            d[0], d[-1] = d[-1], d[0]
            d.rotate()
        return permutations

    @staticmethod
    def _round_robin_odd(d, n):
        permutations = []
        for i in range(n):
            permutations += [[d[j], d[-j - 1]] for j in range(n // 2)]
            d.rotate()
        return permutations

    @staticmethod
    def _round_robin(n):
        d = deque(range(n))
        if n % 2 == 0:
            return list(MergeRunner._round_robin_even(d, n))
        else:
            return list(MergeRunner._round_robin_odd(d, n))

    @staticmethod
    def _min_merger(mergers):
        min_merger = None
        min_distance = 100000000
        for m in mergers:
            if m.min_displacement()[0] < min_distance:
                min_merger = m
                min_distance = m.min_displacement()[0]
        return min_merger

    @staticmethod
    def best_merger(merges):
        invert_dist = not merges[0].minimize  # max value must be first
        return sorted(merges, key=lambda x: np.sum(x.min_distance()), reverse=invert_dist)[0]

    def remove_mergers(self, remove_tag):
        tags_to_remove = []
        for tag in self.candidate_nodes.keys():
            if tag & remove_tag > 0:
                tags_to_remove += [tag]
        for tag in tags_to_remove:
            self.logger.debug('deleting', tag)
            del self.candidate_nodes[tag]

    def mask_ratio(self, tag):
        a = self.images[tag].shape[0] * self.images[tag].shape[1]
        m = np.count_nonzero(self.images[tag][:, :, 3])
        return m / a

    def run(self, merger, dev_idx=-1):
        candidates = [(1 << tag[0], 1 << tag[1]) for tag in self._round_robin(len(self.images))]
        iteration_count = 0
        while len(self.images) > 0:
            self.logger.info('start at', datetime.now())

            for img1_tag, img2_tag in candidates:
                if self.mask_ratio(img1_tag) < self.mask_ratio(img2_tag):
                    target_tag, query_tag = img1_tag, img2_tag
                else:
                    target_tag, query_tag = img2_tag, img1_tag
                self.logger.info('started fitting image {} ({}) to {} ({})'.format(
                    target_tag, self.images[target_tag].shape, query_tag, self.images[query_tag].shape))
                start_date = datetime.now()

                self.candidate_nodes[target_tag | query_tag] = merger(
                    self.images[target_tag], target_tag, dev_idx).merge(
                    self.images[query_tag], query_tag)

                self.logger.info('finished fitting image {} to {}, {}'.format(
                    target_tag, query_tag, datetime.now() - start_date))

            self.logger.info('end at', datetime.now())

            if len(self.candidate_nodes) == 0:
                break

            best_merger = MergeRunner.best_merger(list(self.candidate_nodes.values()))

            self.best_mergers += [best_merger]

            self.logger.debug('min merger with tags', best_merger.target_tag, best_merger.query_tag)

            min_merger_tag = best_merger.target_tag | best_merger.query_tag
            min_image = best_merger.merged_image(weighted=self.weighted)

            self.remove_mergers(min_merger_tag)
            del self.images[best_merger.query_tag]
            del self.images[best_merger.target_tag]

            candidates = [(min_merger_tag, t) for t in self.images.keys()]
            self.logger.debug('adding merged image with tag', min_merger_tag)
            self.images[min_merger_tag] = min_image
            iteration_count += 1
        return self.best_mergers


def compute_descriptors(train_sets, merger):
    descriptors = {}
    start = datetime.now()
    for cls, limbs in train_sets.items():
        print(datetime.now(), cls)
        cls_descriptors = [compute_descriptor_from_patches(patches, merger) for patches in limbs]
        descriptors[cls] = cls_descriptors
    print('Merge Time:', datetime.now() - start)
    return descriptors


def compute_descriptor_from_patches(patches, merger):
    merge_series = MergeRunner(patches, True).run(merger, -1)
    return [merge_series[-1].merged_image()]


def predict(merger, descriptors, sample_sets):
    predictions = []
    start = datetime.now()
    for x in sample_sets:
        cls_pred = {}
        for cls, descs in descriptors.items():
            cls_pred[cls] = [merger(d).merge(t).min_distance() for d, t in zip(descs, x)]
        predictions += [cls_pred]
    print('Prediction Time', datetime.now() - start)
    return predictions
