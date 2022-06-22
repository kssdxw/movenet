const fs = require('fs');
const path = require('path');
const process = require('process');
const log = require('@vladmandic/pilogger');
const tf = require('@tensorflow/tfjs-node');
const canvas = require('canvas');
const { exec } = require('child_process');
const modelOptions = {
  // modelPath: 'file://model-lightning3/movenet-lightning.json',
  modelPath: 'file://model-lightning4/movenet-lightning.json',
  // modelPath: 'file://model-thunder3/movenet-thunder.json',
  // modelPath: 'file://model-thunder4/movenet-thunder.json',
};

const bodyParts = ['nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar', 'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow', 'leftWrist', 'rightWrist', 'leftHip', 'rightHip', 'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle'];
let filedir = '';
// save image with processed results
async function saveImage(res, img,dir) {
  // create canvas
  const c = new canvas.Canvas(img.inputShape[1], img.inputShape[0]);
  const ctx = c.getContext('2d');

  // load and draw original image
  const original = await canvas.loadImage(img.fileName);
  ctx.drawImage(original, 0, 0, c.width, c.height);
  // const fontSize = Math.trunc(c.width / 50);
  const fontSize = Math.round((c.width * c.height) ** (1 / 2) / 80);
  ctx.lineWidth = 2;
  ctx.strokeStyle = 'white';
  ctx.font = `${fontSize}px "Segoe UI"`;

  // draw all detected objects
  for (const obj of res) {
    ctx.fillStyle = 'black';
    ctx.fillText(`${Math.round(100 * obj.score)}% ${obj.label}`, obj.x + 1, obj.y + 1);
    ctx.fillStyle = 'white';
    ctx.fillText(`${Math.round(100 * obj.score)}% ${obj.label}`, obj.x, obj.y);
  }
  ctx.stroke();

  const connectParts = (parts, color) => {
    ctx.strokeStyle = color;
    ctx.beginPath();
    for (let i = 0; i < parts.length; i++) {
      const part = res.find((a) => a.label === parts[i]);
      if (part) {
        if (i === 0) ctx.moveTo(part.x, part.y);
        else ctx.lineTo(part.x, part.y);
      }
    }
    ctx.stroke();
  };

  connectParts(['nose', 'leftEye', 'rightEye', 'nose'], '#99FFFF');
  connectParts(['rightShoulder', 'rightElbow', 'rightWrist'], '#99CCFF');
  connectParts(['leftShoulder', 'leftElbow', 'leftWrist'], '#99CCFF');
  connectParts(['rightHip', 'rightKnee', 'rightAnkle'], '#9999FF');
  connectParts(['leftHip', 'leftKnee', 'leftAnkle'], '#9999FF');
  connectParts(['rightShoulder', 'leftShoulder', 'leftHip', 'rightHip', 'rightShoulder'], '#9900FF');

  // write canvas to jpeg
  if (!fs.existsSync(`${dir}/outputs`)) {
    fs.mkdirSync(`${dir}/outputs`)
  }
  const outImage = `${dir}/outputs/${path.basename(img.fileName)}`;
  const out = fs.createWriteStream(outImage);
  out.on('finish', () => log.state('Created output image:', outImage, 'size:', [c.width, c.height]));
  out.on('error', (err) => log.error('Error creating image:', outImage, err));
  const stream = c.createJPEGStream({ quality: 0.6, progressive: true, chromaSubsampling: true });
  stream.pipe(out);
  if (!fs.existsSync(`${dir}/txt`)) {
    fs.mkdirSync(`${dir}/txt`)
  }
  let result = {"version":1.3,"people":[{"person_id":[-1],"pose_keypoints_2d":[],"face_keypoints_2d":[],"hand_left_keypoints_2d":[],"hand_right_keypoints_2d":[],"pose_keypoints_3d":[],"face_keypoints_3d":[],"hand_left_keypoints_3d":[],"hand_right_keypoints_3d":[]}]}
  res.map((value)=>{
    result.people[0].pose_keypoints_2d.push(value.x)
    result.people[0].pose_keypoints_2d.push(value.y)
    result.people[0].pose_keypoints_2d.push(value.score)
  })
  fs.writeFile(`${dir}/txt/${path.basename(img.fileName).replace(".jpg",".json")}`, JSON.stringify(result), err => {
  if (err) {
    console.error(err)
    return
  }
  //文件写入成功。
})
}

// load image from file and prepares image tensor that fits the model
async function loadImage(fileName, inputSize) {
  const data = fs.readFileSync(fileName);
  const obj = tf.tidy(() => {
    const buffer = tf.node.decodeImage(data);
    const expand = buffer.expandDims(0);
    // @ts-ignore
    const resize = tf.image.resizeBilinear(expand, [inputSize, inputSize]);
    const cast = tf.cast(resize, 'int32');
    const tensor = cast;
    const img = { fileName, tensor, inputShape: buffer?.shape, modelShape: tensor?.shape, size: buffer?.size };
    return img;
  });
  return obj;
}

async function processResults(res, img) {
  const data = res.arraySync();
  log.info('Tensor output', res.shape);
  // log.data(data);
  res.dispose();
  const kpt = data[0][0];
  const parts = [];
  for (let i = 0; i < kpt.length; i++) {
    const part = {
      id: i,
      label: bodyParts[i],
      score: kpt[i][2],
      xRaw: kpt[i][0],
      yRaw: kpt[i][1],
      x: Math.trunc(kpt[i][1] * img.inputShape[1]),
      y: Math.trunc(kpt[i][0] * img.inputShape[0]),
    };
    parts.push(part);
  }
  return parts;
}
async function main(){
  const filelist = process.argv.length > 2 ? process.argv[2] : null;
  var arr = fs.readdirSync(filelist);
 
  arr.forEach(function(item){
    var fullpath = path.join(filelist,item);
    var stats = fs.statSync(fullpath);
    if(!stats.isDirectory()&&fullpath.indexOf('mp4')!=-1){
      if (!fs.existsSync(`${fullpath.replace('.mp4','')}`)) {
        fs.mkdirSync(`${fullpath.replace('.mp4','')}`)
      }
      exec(`ffmpeg -i ${fullpath} -r 30.0 ${fullpath.replace('.mp4','')}/%4d.jpg`, (err, stdout, stderr) => {
          var imglist = fs.readdirSync(fullpath.replace('.mp4',''));
          imglist.forEach(function(image){
            var imagePath = path.join(fullpath.replace('.mp4',''),image);
            var imageStats = fs.statSync(imagePath);
            if(!imageStats.isDirectory()&&imagePath.indexOf('jpg')!=-1){
              domain(imagePath,fullpath.replace('.mp4',''))
            }
          })
      })
      
    }
  });
  

}

async function domain(filename,dir) {
  log.header();

  // init tensorflow
  await tf.enableProdMode();
  await tf.setBackend('tensorflow');
  await tf.ENV.set('DEBUG', false);
  await tf.ready();

  // load model
  const model = await tf.loadGraphModel(modelOptions.modelPath);
  log.info('Loaded model', modelOptions, 'tensors:', tf.engine().memory().numTensors, 'bytes:', tf.engine().memory().numBytes);
  // @ts-ignore
  log.info('Model Signature', model.signature);

  // load image and get approprite tensor for it
  let inputSize = Object.values(model.modelSignature['inputs'])[0].tensorShape.dim[2].size;
  if (inputSize === -1) inputSize = 256;
  const imageFile =filename;
  if (!imageFile || !fs.existsSync(imageFile)) {
    log.error('Specify a valid image file');
    process.exit();
  }
  const img = await loadImage(imageFile, inputSize);
  log.info('Loaded image:', img.fileName, 'inputShape:', img.inputShape, 'modelShape:', img.modelShape, 'decoded size:', img.size);

  // run actual prediction
  const t0 = process.hrtime.bigint();
  // for (let i = 0; i < 99; i++) model.execute(img.tensor); // benchmarking
  const res = model.execute(img.tensor);
  const t1 = process.hrtime.bigint();
  log.info('Inference time:', Math.round(parseInt((t1 - t0).toString()) / 1000 / 1000), 'ms');

  // process results
  const results = await processResults(res, img);
  const t2 = process.hrtime.bigint();
  log.info('Processing time:', Math.round(parseInt((t2 - t1).toString()) / 1000 / 1000), 'ms');

  // print results
  log.data('Results:', results);

  // save processed image
  await saveImage(results, img,dir);
}

main();
