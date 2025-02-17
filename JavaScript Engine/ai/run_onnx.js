import * as ort from 'onnxruntime-web';

async function runModel(inputArray) {
    const session = await ort.InferenceSession.create('/path/to/onnx_model.onnx');

    const inputTensor = new ort.Tensor('float32', new Float32Array(inputArray), [25]);
    const feeds = { input: inputTensor };

    const results = await session.run(feeds);

    const output = results.output.data;
    console.log('Model Output:', output);

    return output;
}

// for dummy data
const testInput = Array(25).fill(0); // replace this part with the actual data
runModel(testInput);