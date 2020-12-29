import numpy as np
import pyopencl as cl


class BaseClassifier:
    def __init__(self, program_file, minimize, descriptors_dict, dev_idx=-1):
        self.platform = cl.get_platforms()[0]
        self.context = cl.Context([self.platform.get_devices()[dev_idx]])
        self.program = cl.Program(self.context, open(program_file).read()).build()
        self.cl_descriptors = {costume: [cl.image_from_array(self.context, de, 4, mode='r') for de in des]
                               for costume, des in descriptors_dict.items()}
        self.queue = cl.CommandQueue(self.context)
        self.minimize = minimize

    def predict(self, sample_set):
        cl_sample_set = [cl.image_from_array(self.context, sample, 4, mode='r')
                         if sample is not None else None for sample in sample_set]

        # predict score of each sample, mean of every color channel summed up over all cutouts
        # numpy's float32 is not serializable, thus, cast to python native float
        return {costume_id: float(np.sum(
            [np.mean(self.distance(des[i // 2], sample)) for i, sample in enumerate(cl_sample_set) if sample is not None]))
            for costume_id, des in self.cl_descriptors.items()}


class MseClassifier(BaseClassifier):
    KERNEL_PATH = 'resources/kernels/mean_squared_error.opencl'

    def __init__(self, descriptors_dict, dev_idx=-1):
        super().__init__(MseClassifier.KERNEL_PATH, True, descriptors_dict, dev_idx)

    def distance(self, cl_target, cl_query):
        out_dims = (cl_query.shape[0] // 4) * 2 + 1, (cl_query.shape[1] // 4) * 2 + 1
        dest = cl.Image(self.context, cl.mem_flags.WRITE_ONLY,
                        cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                        shape=(out_dims[0], out_dims[1], 11))
        self.program.mse(self.queue, dest.shape, None,
                         cl_target, np.array(cl_target.shape, dtype=np.int32),
                         cl_query, np.array(cl_query.shape, dtype=np.int32),
                         dest, np.array(dest.shape[:2], dtype=np.int32)).wait()
        out = np.empty((11, dest.shape[1], dest.shape[0], 4), dtype=np.float32)
        cl.enqueue_copy(self.queue, out, dest, origin=(0, 0), region=dest.shape)
        distance_space = out[:, :, :, :3]

        reduced_distance_space = np.nansum(distance_space, axis=3)
        rot_i, ty_i, tx_i = np.unravel_index(np.nanargmin(reduced_distance_space), reduced_distance_space.shape)
        return distance_space[rot_i, ty_i, tx_i]

    def __str__(self):
        return 'Mean Squared Error classifier, number of classes: %s' % len(self.cl_descriptors)
        

class CcClassifier(BaseClassifier):
    KERNEL_PATH = 'resources/kernels/correlation_coefficient.opencl'

    def __init__(self, descriptors_dict, dev_idx=-1):
        super().__init__(CcClassifier.KERNEL_PATH, False, descriptors_dict, dev_idx)

    def distance(self, cl_target, cl_query):
        out_dims = (cl_query.shape[0] // 4) * 2 + 1, (cl_query.shape[1] // 4) * 2 + 1
        img1avg = cl.Image(self.context, cl.mem_flags.READ_WRITE,
                           cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                           shape=(out_dims[0], out_dims[1], 11))
        img2avg = cl.Image(self.context, cl.mem_flags.READ_WRITE,
                           cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                           shape=(out_dims[0], out_dims[1], 11))
        corrco = cl.Image(self.context, cl.mem_flags.WRITE_ONLY,
                          cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT),
                          shape=(out_dims[0], out_dims[1], 11))
        self.program.avg(self.queue, img1avg.shape, None,
                         cl_target, np.array(cl_target.shape, dtype=np.int32),
                         cl_query, np.array(cl_query.shape, dtype=np.int32),
                         img1avg, img2avg, np.array(img1avg.shape[:2], dtype=np.int32)).wait()
        self.program.corr(self.queue, img1avg.shape, None,
                          cl_target, np.array(cl_target.shape, dtype=np.int32),
                          cl_query, np.array(cl_query.shape, dtype=np.int32),
                          img1avg, img2avg, corrco, np.array(img1avg.shape[:2], dtype=np.int32)).wait()
        out = np.empty((11, img2avg.shape[1], img2avg.shape[0], 4), dtype=np.float32)
        cl.enqueue_copy(self.queue, out, corrco, origin=(0, 0), region=img2avg.shape)
        distance_space = out[:, :, :, :3]

        reduced_distance_space = np.nansum(distance_space, axis=3)
        rot_i, ty_i, tx_i = np.unravel_index(np.nanargmax(reduced_distance_space), reduced_distance_space.shape)
        return distance_space[rot_i, ty_i, tx_i]

    def __str__(self):
        return 'Correlation Coefficient classifier, number of classes: %s' % len(self.cl_descriptors)
