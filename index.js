const detectImage = async (
  image,
  session,
  topk,
  iouThreshold,
  scoreThreshold,
  inputShape
) => {
  var canvas = document.createElement("canvas");

  canvas.width = 640;
  canvas.height = 640;
  canvas.id = "canvas";

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height); // clean canvas

  const [modelWidth, modelHeight] = inputShape.slice(2);
  const maxSize = Math.max(modelWidth, modelHeight); // max size in input model

  const inputTensors = [];
  var Ratio = [];

  for (let i = 0; i < image.length; i++) {
    const [input, xRatio, yRatio] = preprocessing(
      image[i],
      modelWidth,
      modelHeight
    );

    const transposedInput = input.transpose([0, 3, 1, 2]);
    const inputData = transposedInput.dataSync();
    const expectedSize = inputShape.reduce((a, b) => a * b, 1);
    if (inputData.length !== expectedSize) {
      throw new Error(
        `Data length (${inputData.length}) does not match expected size (${expectedSize})`
      );
    }
    const float32InputData = new Float32Array(
      inputData.buffer,
      inputData.byteOffset,
      inputData.byteLength / Float32Array.BYTES_PER_ELEMENT
    );
    if (float32InputData.length !== expectedSize) {
      throw new Error(
        `Float32 data length (${float32InputData.length}) does not match expected size (${expectedSize})`
      );
    }
    const inputCopy = new Float32Array(float32InputData);

    // Create ort.Tensor with Float32Array
    const tensor = new ort.Tensor("float32", inputCopy, inputShape);
    Ratio[i] = [xRatio, yRatio];

    inputTensors.push(tensor);
  }

  const config = new ort.Tensor(
    "float32",
    new Float32Array([
      80, // num class
      topk, // topk per class
      iouThreshold, // iou threshold
      scoreThreshold, // score threshold
    ])
  );

  let concatSize = 0;
  inputTensors.forEach((tensor) => {
    concatSize += tensor.size;
  });

  const concatenatedData = new Float32Array(concatSize);

  let index = 0;
  inputTensors.forEach((tensor) => {
    concatenatedData.set(tensor.data, index);
    index += tensor.size;
  });

  const concatenatedTensor = new ort.Tensor("float32", concatenatedData, [
    image.length,
    3,
    640,
    640,
  ]);
  const { output0, output1 } = await session.net.run({
    images: concatenatedTensor,
  });

  // const feeds = {
  //   input: concatenatedTensor,
  // };
  
  // const { output0, output1 } = await session.net.run(feeds);
  console.log("output0",output0);
  console.log("output1",output1);


  var arrayOutput0 = [];
  const Data0 = output0.data;
  const numImages = image.length;
  const PerImage = Data0.length / numImages;

  var arrayOutput1 = [];
  const maskData = output1.data;
  const masksPerImage = maskData.length / numImages;


  for(var i=0;i<numImages;i++){
    const startIdx = i * PerImage;
    const endIdx = startIdx + PerImage;
    const ForImage = Data0.slice(startIdx, endIdx);
    const output00 = {
      dims: [1, 116, 8400],
      size: 974400, 
      type: "float32",
      data: ForImage
      
    };
    arrayOutput0.push(output00);


    const startIdx1 = i * masksPerImage;
    const endIdx1 = startIdx1 + masksPerImage;
    const maskForImage = maskData.slice(startIdx1, endIdx1);
    const output11 = {
      dims: [1, 32,160, 160],
      size: 819200, 
      type: "float32",
      data: maskForImage
      
    };
    arrayOutput1.push(output11);
              
  }

  let indexImage=0; 
  drawingMasksOnArrayImages(indexImage,arrayOutput1,arrayOutput0, session , image , maxSize , config , modelHeight,modelWidth,Ratio);
};

function runInference() {
  let img = document.querySelectorAll("img");
  img = Array.from(img);
  detectImage(
    img,
    mySession,
    topk,
    iouThreshold,
    scoreThreshold,
    modelInputShape
  );
}

if (!OnnxLoaded) {
  var idInterval = setInterval(function () {
    if (OnnxLoaded) {
      runInference();
      clearInterval(idInterval);
    }
  }, 1000);
} else {
  runInference();
}

async function draw_apart(
  boxes,
  arrayOutput1,
  selected,
  i,
  maxSize,
  Ratio,
  session,
  overlay
) {
  for (let idx = 0; idx < selected.dims[1]; idx++) {
    const data = selected.data.slice(
      idx * selected.dims[2],
      (idx + 1) * selected.dims[2]
    );
    let box = data.slice(0, 4);
    const scores = data.slice(4, 4 + numClass);
    const score = Math.max(...scores);
    const label = scores.indexOf(score);
    const color = colors.get(label);
    if (labels[label] == "person") {
      box = overflowBoxes(
        [box[0] - 0.5 * box[2], box[1] - 0.5 * box[3], box[2], box[3]],
        maxSize
      );

      const [x, y, w, h] = overflowBoxes(
        [
          Math.floor(box[0] * Ratio[i][0]),
          Math.floor(box[1] * Ratio[i][1]),
          Math.floor(box[2] * Ratio[i][0]),
          Math.floor(box[3] * Ratio[i][1]),
        ],
        maxSize
      );

      boxes.push({
        label: labels[label],
        probability: score,
        color: color,
        bounding: [x, y, w, h],
      });

      const mask = new ort.Tensor(
        "float32",
        new Float32Array([...box, ...data.slice(4 + numClass)])
      );
      const maskConfig = new ort.Tensor(
        "float32",
        new Float32Array([maxSize, x, y, w, h, ...Colors.hexToRgba(color, 120)])
      );
      const { mask_filter } = await session.mask.run({
        detection: mask,
        mask: arrayOutput1[i],
        config: maskConfig,
      });
      const mask_mat = cv.matFromArray(
        mask_filter.dims[0],
        mask_filter.dims[1],
        cv.CV_8UC4,
        mask_filter.data
      ); // mask result to Mat

      cv.addWeighted(overlay, 1, mask_mat, 1, 0, overlay); // Update mask overlay
    }
  }
}

async function drawingMasksOnArrayImages(i,arrayOutput1,arrayOutput0, session ,image,maxSize,config ,modelHeight,modelWidth , Ratio) {
  var img = image[i];
  let canvas = document.createElement("canvas");
  canvas.width = 640;
  canvas.height = 640;
  canvas.id = "canvas";
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  const { selected } = await session.nms.run({
    detection: arrayOutput0[i],
    config: config,
  });

  const boxes = [];

  const overlay = cv.Mat.zeros(modelHeight, modelWidth, cv.CV_8UC4);

  await draw_apart(
    boxes,
    arrayOutput1,
    selected,
    i,
    maxSize,
    Ratio,
    session,
    overlay
  );

  const mask_img = new ImageData(
    new Uint8ClampedArray(overlay.data),
    overlay.cols,
    overlay.rows
  );

  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const newData = new Uint8ClampedArray(
    imageData.data.map((value, index) => {
      if (mask_img.data[index % mask_img.data.length] >= 50) {
        switch (index % 4) {
          case 0:
            return 100;
          case 1:
            return 100;
          case 2:
            return 100;
          case 3:
            return 100;
          default:
            return value;
        }
      }
      return value;
    })
  );
  imageData.data.set(newData);
  ctx.putImageData(imageData, 0, 0);

  var can = document.createElement("canvas");
  can.width = img.width;
  can.height = img.height;
  var ctxcan = can.getContext("2d");
  ctxcan.drawImage(canvas, 0, 0, img.width, img.height);
  img.src = can.toDataURL("image/png");
   
  if( i<arrayOutput0.length-1)
    {
        i++;
        await drawingMasksOnArrayImages(i,arrayOutput1,arrayOutput0, session ,image,maxSize,config ,modelHeight,modelWidth , Ratio)
    }
    else {
        return;
    }
}
