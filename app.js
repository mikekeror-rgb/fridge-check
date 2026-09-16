const CLASSES = ["Apple", "Banana", "Orange", "Tomato", "Egg (Food)", "Milk", "Cheese", "Bottle", "Tin can"];
const ESSENTIALS = ["Milk", "Egg (Food)", "Apple", "Bottle"]; // customize your "should always have" list

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const resultsDiv = document.getElementById("results");
const captureBtn = document.getElementById("captureBtn");
const checkMissing = document.getElementById("checkMissing");

let session;

async function init() {
  // Ask for camera access
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
  video.srcObject = stream;

  // Load the ONNX model
  session = await ort.InferenceSession.create("best.onnx");
}

function preprocess(imageData) {
  const { data, width, height } = imageData;
  const floatData = new Float32Array(3 * 640 * 640);
  for (let i = 0; i < 640 * 640; i++) {
    floatData[i] = data[i * 4] / 255;                     // R
    floatData[640 * 640 + i] = data[i * 4 + 1] / 255;      // G
    floatData[2 * 640 * 640 + i] = data[i * 4 + 2] / 255;  // B
  }
  return new ort.Tensor("float32", floatData, [1, 3, 640, 640]);
}

function nms(boxes, scores, iouThreshold) {
  const indices = scores.map((s, i) => i).sort((a, b) => scores[b] - scores[a]);
  const keep = [];
  while (indices.length > 0) {
    const current = indices.shift();
    keep.push(current);
    for (let i = indices.length - 1; i >= 0; i--) {
      const iou = boxIoU(boxes[current], boxes[indices[i]]);
      if (iou > iouThreshold) indices.splice(i, 1);
    }
  }
  return keep;
}

function boxIoU(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const areaA = (a[2] - a[0]) * (a[3] - a[1]);
  const areaB = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (areaA + areaB - inter);
}

function postprocess(output, confThreshold = 0.3) {
  // YOLOv8 ONNX output shape: [1, 4+numClasses, 8400]
  const data = output.data;
  const numBoxes = output.dims[2];
  const numClasses = CLASSES.length;

  const boxes = [], scores = [], classIds = [];

  for (let i = 0; i < numBoxes; i++) {
    let maxScore = 0, maxClass = -1;
    for (let c = 0; c < numClasses; c++) {
      const score = data[(4 + c) * numBoxes + i];
      if (score > maxScore) { maxScore = score; maxClass = c; }
    }
    if (maxScore > confThreshold) {
      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w = data[2 * numBoxes + i];
      const h = data[3 * numBoxes + i];
      boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
      scores.push(maxScore);
      classIds.push(maxClass);
    }
  }

  const keepIndices = nms(boxes, scores, 0.45);
  return keepIndices.map(i => ({ class: CLASSES[classIds[i]], score: scores[i] }));
}

async function scan() {
  ctx.drawImage(video, 0, 0, 640, 640);
  const imageData = ctx.getImageData(0, 0, 640, 640);
  const tensor = preprocess(imageData);

  const feeds = { images: tensor };
  const output = await session.run(feeds);
  const detections = postprocess(output[Object.keys(output)[0]]);

  renderResults(detections);
}

function renderResults(detections) {
  const foundClasses = new Set(detections.map(d => d.class));

  let html = "<h3>Detected:</h3>";
  if (detections.length === 0) {
    html += "<p>Nothing detected — try again with better lighting.</p>";
  }
  for (const d of detections) {
    html += `<div class="item"><span>${d.class}</span><span>${(d.score * 100).toFixed(0)}%</span></div>`;
  }

  if (checkMissing.checked) {
    const missing = ESSENTIALS.filter(e => !foundClasses.has(e));
    if (missing.length > 0) {
      html += `<h3>Missing essentials:</h3>`;
      for (const m of missing) {
        html += `<div class="item missing">${m}</div>`;
      }
    } else {
      html += `<p>All essentials present ✅</p>`;
    }
  }

  resultsDiv.innerHTML = html;
}

captureBtn.addEventListener("click", scan);
init();