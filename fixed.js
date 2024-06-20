let mySession, OnnxLoaded;
const topk = 100;
const iouThreshold = 0.45;
const scoreThreshold = 0.2;
const labels = [
  "person",
  "bicycle",
  "car",
  "motorcycle",
  "airplane",
  "bus",
  "train",
  "truck",
  "boat",
  "traffic light",
  "fire hydrant",
  "stop sign",
  "parking meter",
  "bench",
  "bird",
  "cat",
  "dog",
  "horse",
  "sheep",
  "cow",
  "elephant",
  "bear",
  "zebra",
  "giraffe",
  "backpack",
  "umbrella",
  "handbag",
  "tie",
  "suitcase",
  "frisbee",
  "skis",
  "snowboard",
  "sports ball",
  "kite",
  "baseball bat",
  "baseball glove",
  "skateboard",
  "surfboard",
  "tennis racket",
  "bottle",
  "wine glass",
  "cup",
  "fork",
  "knife",
  "spoon",
  "bowl",
  "banana",
  "apple",
  "sandwich",
  "orange",
  "broccoli",
  "carrot",
  "hot dog",
  "pizza",
  "donut",
  "cake",
  "chair",
  "couch",
  "potted plant",
  "bed",
  "dining table",
  "toilet",
  "tv",
  "laptop",
  "mouse",
  "remote",
  "keyboard",
  "cell phone",
  "microwave",
  "oven",
  "toaster",
  "sink",
  "refrigerator",
  "book",
  "clock",
  "vase",
  "scissors",
  "teddy bear",
  "hair drier",
  "toothbrush",
];
const useState = (defaultValue) => {
  let value = defaultValue;
  const getValue = () => value;
  const setValue = (newValue) => (value = newValue);
  return [getValue, setValue];
};
const numClass = labels.length;
const [session, setSession] = useState(null);

const modelInputShape = [1, 3, 640, 640];

cv["onRuntimeInitialized"] = async () => {
  const [yolo, nms, mask] = await Promise.all([
    ort.InferenceSession.create(
      "yolov8-seg-batching.onnx"
      // ,{executionProviders: ["webgpu"]}
    ),
    ort.InferenceSession.create("nms-yolov8.onnx"),
    ort.InferenceSession.create("mask-yolov8-seg.onnx"),
  ]);
  mySession = setSession({ net: yolo, nms: nms, mask: mask });
  OnnxLoaded = true;
};

const preprocessing = (source, modelWidth, modelHeight, stride = 32) => {
  // Load the image using TensorFlow.js
  const img = tf.browser.fromPixels(source);

  // Resize the image to the desired model input size
  const resizedImg = tf.image.resizeBilinear(img, [modelHeight, modelWidth]);

  // Convert image to 0-1 range
  const normalizedImg = resizedImg.div(255); // should be 255 to normalize to 0-1 range

  // Calculate padding dimensions to make the image square
  const [height, width] = normalizedImg.shape.slice(0, 2);
  const maxLength = Math.max(height, width);
  const topPad = Math.floor((maxLength - height) / 2);
  const bottomPad = maxLength - height - topPad;
  const leftPad = Math.floor((maxLength - width) / 2);
  const rightPad = maxLength - width - leftPad;

  // Pad the image with zeros to make it square
  const paddedImg = tf.pad(normalizedImg, [
    [topPad, bottomPad],
    [leftPad, rightPad],
    [0, 0],
  ]);

  // Expand the dimensions to match the expected input shape
  const input = paddedImg.expandDims();

  // Compute the padding ratios
  const xRatio = modelWidth / normalizedImg.shape[1];
  const yRatio = modelHeight / normalizedImg.shape[0];
  // console.log("[input, xRatio, yRatio]", [input, xRatio, yRatio]);
  return [input, xRatio, yRatio];
};

/**
 * Handle overflow boxes based on maxSize
 * @param {Number[4]} box box in [x, y, w, h] format
 * @param {Number} maxSize
 * @returns non overflow boxes
 */
const overflowBoxes = (box, maxSize) => {
  box[0] = box[0] >= 0 ? box[0] : 0;
  box[1] = box[1] >= 0 ? box[1] : 0;
  box[2] = box[0] + box[2] <= maxSize ? box[2] : maxSize - box[0];
  box[3] = box[1] + box[3] <= maxSize ? box[3] : maxSize - box[1];
  return box;
};
class Colors {
  // ultralytics color palette https://ultralytics.com/
  constructor() {
    this.palette = [
      "#FF3838",
      "#FF9D97",
      "#FF701F",
      "#FFB21D",
      "#CFD231",
      "#48F90A",
      "#92CC17",
      "#3DDB86",
      "#1A9334",
      "#00D4BB",
      "#2C99A8",
      "#00C2FF",
      "#344593",
      "#6473FF",
      "#0018EC",
      "#8438FF",
      "#520085",
      "#CB38FF",
      "#FF95C8",
      "#FF37C7",
    ];
    this.n = this.palette.length;
  }

  get = (i) => this.palette[Math.floor(i) % this.n];

  static hexToRgba = (hex, alpha) => {
    let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? [
          parseInt(result[1], 16),
          parseInt(result[2], 16),
          parseInt(result[3], 16),
          alpha,
        ]
      : null;
  };
}
const colors = new Colors();
