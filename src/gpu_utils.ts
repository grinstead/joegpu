// prettier-ignore
export const QUAD_VERTICES = [
  -1, -1,
   1, -1,
  -1,  1,
   1,  1,
];

export function writeToBuffer(
  queue: GPUQueue,
  buffer: GPUBuffer,
  data: Uint8Array | Uint32Array | Float32Array
) {
  queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
}
