# coding: utf-8
"""
MTCNN 的检测程序
"""
import cv2
import numpy as np
import time
from Caffe2Predictor import py_nms, SpatialSoftmax
from pylab import plt



def square_bbox(bbox):
    """
    把bbox，变成方形（以bbox的中心为中心，最长边为边）
    :param bbox:
    :return:
    """
    h = bbox[3]
    w = bbox[2]
    max_size = np.max((w,h))

    sq = bbox.copy()
    sq[0] = bbox[0] + w * 0.5 - max_size * 0.5
    sq[1] = bbox[1] + h * 0.5 - max_size * 0.5
    sq[2] = max_size
    sq[3] = max_size
    return sq

def pad_bbox(bbox, W, H):
    """
    计算bbox在原图中的坐标，及拷贝到小图的坐标
    :param bbox:
    :return:
    """
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2] + x0
    y1 = bbox[3] + y0

    # src dst
    sx0 = x0
    dx0 = 0
    sy0 = y0
    dy0 = 0
    sx1 = x1
    dx1 = bbox[2]
    sy1 = y1
    dy1 = bbox[3]

    # 如果x小于0
    if x0 < 0:
        sx0 = 0
        dx0 = -x0
    if y0 < 0:
        sy0 = 0
        dy0 = -y0

    if x1 > W - 1:
        sx1 = W - 1
        dx1 = sx1 - sx0
    if y1 > H - 1:
        sy1 = H - 1
        dy1 = sy1 - sy0

    #
    miny = np.min((dy1-dy0, sy1-sy0))
    dy1 = dy0 + miny
    sy1 = sy0 + miny
    minx = np.min((dx1-dx0, sx1-sx0))
    dx1 = dx0 + minx
    sx1 = sx0 + minx

    return sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1




class MTCNN(object):

    def __init__(self,
                 detectors,
                 min_face_size=40,
                 stride=2,
                 threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.79):

        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor

    def generate_bbox(self, cls_map, reg, scale, threshold):
        """
            根据cls_map（预测结果图）与thread，选择相应的回归框，并将其映射回原图
        Parameters:
        ----------
            cls_map: numpy array , 2*h*w
                detect score for each position
            reg: numpy array , 4*h*w
                reg bbox
            scale: float number
                scale of this image pyramid from original image
            threshold: float number
                detect threshold
        Returns:
        -------
            bbox array in original image， num*5, [x,y,w,h,score]
        """
        stride = 2
        cellsize = 12
        face_map = cls_map[1, :, :]
        t_index = np.where(face_map > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])
        # offset
        # dx, dy, dw, dh = [reg[t_index[0], t_index[1], i] for i in range(4)]
        dx, dy, dw, dh = [reg[i, t_index[0], t_index[1]] for i in range(4)]

        ## backwark for smooth l1 loss (RCNN)
        dx *= cellsize
        dy *= cellsize
        dw = np.exp(dw) * cellsize
        dh = np.exp(dh) * cellsize

        # 各回归框的分数
        score = face_map[t_index[0], t_index[1]]

        # 加上Gx, Gy，映射回原图
        Gx = np.round(stride * t_index[1] / scale)
        Gy = np.round(stride * t_index[0] / scale)
        dx = dx / scale + Gx
        dy = dy / scale + Gy
        dw = dw / scale
        dh = dh / scale
        # 组合结果
        bbox = np.vstack([dx, dy, dw, dh, score])
        bbox = bbox.T
        return bbox

    def detect_pnet(self, im):
        """Get face candidates through pnet

        Parameters:
        ----------
        im: numpy array
            input image array

        Returns:
        -------
        boxes: numpy array
            detected boxes before calibration
        boxes_c: numpy array
            boxes after calibration
        """

        # 计算需要做多少次scale
        h, w, c = im.shape
        net_size = 12
        minl = np.min((w, h))
        base_scale = net_size / float(self.min_face_size)
        scales = []
        face_count = 0
        while minl > net_size:
            s = base_scale * self.scale_factor ** face_count
            if np.floor(minl * s) <= 12:
                break
            scales += [s]
            face_count += 1

        # 对每个scale层做预测
        total_boxes = []
        for scale in scales:
            hs = np.ceil(h * scale)
            ws = np.ceil(w * scale)
            hs = int(hs)
            ws = int(ws)
            im_data = cv2.resize(im, (ws, hs))

            # predict
            out_blobs = self.pnet_detector.predict(im_data, ["conv4_1", "conv4_2"])

            # 二分类预测
            cls_map = SpatialSoftmax(out_blobs[0])[0, :, :, :]
            # 边框回归
            reg = out_blobs[1][0, :, :, :]

            # 生成边界框（映射回原图）
            bboxes = self.generate_bbox(cls_map, reg, scale, self.thresh[0])

            # inter-scale nms
            if len(bboxes) <= 0:
                continue
            keep = py_nms(bboxes, 0.5, 'Union')

            if len(keep) <= 0:
                continue

            bboxes = bboxes[keep]
            #
            total_boxes.extend(bboxes)

        # 金字塔所有层做完之后，再做一次NMS
        # NMS
        if len(total_boxes) <= 0:
            return None
        total_boxes = np.array(total_boxes)
        keep = py_nms(total_boxes, 0.7, 'Union')
        if len(keep) <= 0:
            return None
        return total_boxes[keep]

    def detect_ronet(self, im, bbox, image_size, nms_type, output_blobs, detector):
        """
        检测rnet
        :param im: 原图
        :param bbox: pnet预测的bbox
        :param image_size: 网络输入图像的尺寸，24 or 48
        :param nms_type: 最终结果NMS的类型，Union or Minimum
        :return: 预测的bbox
        """
        H,W,C = im.shape
        IMAGE_SIZE = image_size
        # 1, 先将bbox转换成矩形
        sb = []
        for i in range(bbox.shape[0]):
            box = bbox[i, :]
            sq = square_bbox(box)
            sb += [sq]

        # 2，pad
        crops = []
        origin_bbox = []
        for i in sb:
            size = i[2]
            sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1 = pad_bbox(i, W, H)
            crop = np.zeros((size, size, 3), dtype=np.uint8)
            if sx0 < 0 or sy0 < 0 or dx0 < 0 or dy0 < 0 or sx1 > W or sy1 > H or dx1 > size or dy1 > size:
                continue
            crop[dy0:dy1, dx0:dx1, :] = im[sy0:sy1, sx0:sx1, :]
            out = cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))
            out = out.swapaxes(1,2).swapaxes(0,1)
            crops += [out]
            origin_bbox += [i]

        # 3, 预测
        origin_bbox = np.array(origin_bbox)
        crops = np.array(crops)
        out = detector.predict2(crops, output_blobs)

        # 4，映射
        ## out[0] -> N * 2
        ## out[1] -> N * 4
        cls_map = out[0]
        reg = out[1]

        face_map = cls_map[:,1]
        t_index = np.where(face_map > self.thresh[1])
        if t_index[0].shape[0] <= 0:
            return None

        # 5, NMS
        origin_bbox = origin_bbox[t_index]
        score = face_map[t_index]
        reg_map = reg[t_index]
        origin_score = np.zeros((score.shape[0], 5))
        origin_score[:,0:4] = origin_bbox
        origin_score[:,4] = score

        keep = py_nms(origin_score, 0.6)

        if len(keep) <= 0:
            return None
        reg_map = reg_map[keep]
        origin_bbox = origin_bbox[keep]
        score = score[keep]

        dx = reg_map[:,0]
        dy = reg_map[:,1]
        dw = reg_map[:,2]
        dh = reg_map[:,3]

        # backward for smooth l1 loss(RCNN)
        dx *= IMAGE_SIZE
        dy *= IMAGE_SIZE
        dw = np.exp(dw) * IMAGE_SIZE
        dh = np.exp(dh) * IMAGE_SIZE

        # add Gx AND Gy
        G = origin_bbox
        G = G.astype(np.float32)

        dx = dx / (float(IMAGE_SIZE) / G[:,2]) + G[:,0]
        dy = dy / (float(IMAGE_SIZE) / G[:,3]) + G[:,1]
        dw = dw / (float(IMAGE_SIZE) / G[:,2])
        dh = dh / (float(IMAGE_SIZE) / G[:,3])

        # compose
        bbox = np.vstack([dx, dy, dw, dh, score])
        bbox = bbox.T

        if image_size == 48:
            keep = py_nms(bbox, 0.6, "Minimum")
            if len(keep) <= 0:
                return None
            return bbox[keep]
        else:
            return bbox


    def detect(self, img):
        """
        Detect face over image
        """
        bboxes = None
        # pnet
        if self.pnet_detector:
            bboxes = self.detect_pnet(img)

        if bboxes is None:
            return None

        ## 可视化PNET的结果
        # plt.figure()
        # tmp = img.copy()
        # for i in bboxes:
        #    x0 = int(i[0])
        #    y0 = int(i[1])
        #    x1 = x0 + int(i[2])
        #    y1 = y0 + int(i[3])
        #    cv2.rectangle(tmp, (x0,y0), (x1,y1), (0,0,255), 2)
        # plt.imshow(tmp[:,:,::-1])
        # plt.title("pnet result")

        # rnet
        if self.rnet_detector:
            bboxes = bboxes.astype(np.int32)
            bboxes = bboxes[:,0:4]
            bboxes = self.detect_ronet(img, bboxes, 24, "Union", ["prob_cls", "fc5_2"], self.rnet_detector)
        if bboxes is None:
            return None

        # 可视化Rnet的结果
        # plt.figure()
        # tmp = img.copy()
        # for i in bboxes:
        #    x0 = int(i[0])
        #    y0 = int(i[1])
        #    x1 = x0 + int(i[2])
        #    y1 = y0 + int(i[3])
        #    cv2.rectangle(tmp, (x0, y0), (x1, y1), (0, 0, 255), 2)
        # plt.imshow(tmp[:, :, ::-1])
        # plt.title("rnet result")

        # onet
        if self.onet_detector:
            bboxes = bboxes.astype(np.int32)
            bboxes = bboxes[:,0:4]
            bboxes = self.detect_ronet(img, bboxes, 48, "Minimum", ["prob_cls", "fc6_2"], self.onet_detector)
        return bboxes


if __name__ == "__main__":
    from Caffe2Predictor import C2Predictor
    from caffe2.python import core
    from caffe2.proto import caffe2_pb2

    core.GlobalInit(["MTCNN", "--caffe2_log_level=0"])
    do = core.DeviceOption(caffe2_pb2.CPU, 0)
    # pnet
    pnet_init_path = r"model/models/pnet_20180815_init_net_final.pb"
    pnet_predict_path = r"model/models/pnet_20180815_predict_net.pb"
    pnet_det = C2Predictor(pnet_init_path, pnet_predict_path, 127.5, 0.0078125, do)

    # rnet
    rnet_init_path = r"model/models/rnet_20180815_init_net_final.pb"
    rnet_predict_path = r"model/models/rnet_20180815_predict_net.pb"
    rnet_det = C2Predictor(rnet_init_path, rnet_predict_path, 127.5, 0.0078125, do)
    # onet
    onet_init_path = r"model/models/onet_20180815_init_net_final.pb"
    onet_predict_path = r"model/models/onet_20180815_predict_net.pb"
    onet_det = C2Predictor(onet_init_path, onet_predict_path, 127.5, 0.0078125, do)
    # detector
    detector = [pnet_det, rnet_det, onet_det]
    mt = MTCNN(detector, 40)

    # img = cv2.imread("/home/beichen2012/dataset/faces2.jpg")
    img = cv2.imread(r"d:/1.jpg")
    bbox = mt.detect(img)
    if bbox is not None:
        rows, _ = bbox.shape
        for i in range(rows):
            cv2.rectangle(img, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][0] + bbox[i][2]), int(bbox[i][1] + bbox[i][3])), (0,0,255), 2)
    from pylab import plt
    plt.figure()
    plt.imshow(img[:,:,::-1])
    plt.title("result")
    plt.show()
