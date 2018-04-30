import os
import argparse
import numpy as np
import cntk

from FasterRCNN.FasterRCNN_train import store_eval_model_with_native_udf

from FasterRCNN.FasterRCNN_eval import FasterRCNN_Evaluator
from utils.annotations.annotations_helper import parse_class_map_file
from utils.config_helpers import merge_configs
import utils.od_utils as od


def get_configuration():
    from FasterRCNN.FasterRCNN_config import cfg as detector_cfg
    from utils.configs.VGG16_config import cfg as network_cfg
    from utils.configs.MF_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run predictions.')
    parser.add_argument('-m', '--model', help="CNTK model file", default=r"C:\CNTK\CNTK\Examples\Image\Detection\FasterRCNN\Output\faster_rcnn_eval_VGG16_e2e.model")
    parser.add_argument('-i', '--image', help="Input image file", default=r"C:\Users\Sulej\Desktop\castelli.jpg")
    args = parser.parse_args()
    model_path = args.model
    img_path = args.image

    cfg = get_configuration()
    cfg['MODEL_PATH'] = model_path
    cfg["DATA"].CLASS_MAP_FILE = os.path.join(cfg["DATA"].MAP_FILE_PATH, cfg["DATA"].CLASS_MAP_FILE)
    cfg["DATA"].CLASSES = parse_class_map_file(cfg["DATA"].CLASS_MAP_FILE)
    cfg.NUM_CHANNELS = 3

    cntk.device.try_set_default_device(cntk.device.gpu(cfg.GPU_ID))

    model = cntk.ops.functions.load_model(model_path)

    #store_eval_model_with_native_udf(model, cfg)

    evaluator = FasterRCNN_Evaluator(model, cfg)
    regressed_rois, cls_probs = evaluator.process_image(img_path)

    print("RESULTS_NMS_THRESHOLD = ", cfg.RESULTS_NMS_THRESHOLD, "RESULTS_NMS_CONF_THRESHOLD = ", cfg.RESULTS_NMS_CONF_THRESHOLD)
    bboxes, labels, scores = od.filter_results(regressed_rois, cls_probs, cfg)

    fg_boxes = np.where(labels > 0)
    print("#bboxes: before nms: {}, after nms: {}, foreground: {}".format(len(regressed_rois), len(bboxes), len(fg_boxes[0])))

    od.visualize_results(img_path, bboxes, labels, scores, cfg)

    print("done")