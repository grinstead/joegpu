import { UNIFORM_ALIGNMENT, writeToBuffer } from "./gpu_utils";
import { parallelPrefixSum } from "./parallelPrefixSum";
import { NUM_BYTES_UINT32 } from "./utils";

export async function testParallelPrefixSum(device: GPUDevice) {
  const maxTest = 1 << 20;
  const maxOffset = 10;
  const fullBuffer = new Uint32Array(
    maxTest + maxOffset * (UNIFORM_ALIGNMENT / NUM_BYTES_UINT32)
  );

  const buffer = device.createBuffer({
    label: "Test Values",
    size: maxTest * NUM_BYTES_UINT32,
    usage:
      GPUBufferUsage.COPY_DST |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.STORAGE,
  });

  const outputBuffer = device.createBuffer({
    label: "Test Values - Output",
    size: maxTest * NUM_BYTES_UINT32,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });

  const slice = device.createBuffer({
    label: "Slice Data",
    size: 2 * NUM_BYTES_UINT32,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.UNIFORM,
  });

  outerLoop: for (
    let numElements = 1;
    numElements < maxTest;
    numElements *= 4
  ) {
    console.log(`Testing for ${numElements}-${2 * numElements} elements`);

    const shader = parallelPrefixSum(device, 2 * numElements);
    shader.prep({
      values: buffer,
      slice,
    });

    for (let i = 0; i < 4; i++) {
      const offset =
        (UNIFORM_ALIGNMENT / NUM_BYTES_UINT32) *
        Math.floor(maxOffset * Math.random());
      const total = 2 + Math.floor((i / 3 + Math.random()) * numElements);
      for (let j = 0; j < total; j++) {
        fullBuffer[offset + j] = 8 * Math.random() ** 2;
      }

      console.log(`Running ${total}`);

      writeToBuffer(
        device.queue,
        buffer,
        fullBuffer.subarray(0, offset + total)
      );
      writeToBuffer(device.queue, slice, new Uint32Array([offset, total]));

      const encoder = device.createCommandEncoder();

      shader.runPrefixSum(encoder, total);

      encoder.copyBufferToBuffer(
        buffer,
        offset * NUM_BYTES_UINT32,
        outputBuffer,
        0,
        total * NUM_BYTES_UINT32
      );

      device.queue.submit([encoder.finish()]);

      console.log(`Running! {offset: ${offset}, length: ${total}}`);
      if (total < 10) {
        console.log(
          `[${fullBuffer.subarray(offset, offset + total).join(", ")}]`
        );
      }

      await device.queue.onSubmittedWorkDone();

      console.log("Finished! Retrieving Results");

      await outputBuffer.mapAsync(GPUMapMode.READ, 0, total * NUM_BYTES_UINT32);

      const results = new Uint32Array(
        outputBuffer.getMappedRange(0, total * NUM_BYTES_UINT32)
      );

      console.log("Comparing...");

      let sum = 0;
      for (let j = 0; j < total; j++) {
        if (results[j] !== sum) {
          console.error(
            `Error! results[${j}] expected ${sum} got ${results[j]}`
          );

          console.log(
            "input",
            Array.from(fullBuffer.subarray(offset, offset + total))
          );
          console.log("output", Array.from(results.subarray(0, total)));

          break outerLoop;
        }

        sum += fullBuffer[offset + j];
      }

      console.log(`Success!`);

      outputBuffer.unmap();
    }
  }
}
