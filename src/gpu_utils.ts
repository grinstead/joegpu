// prettier-ignore
export const QUAD_VERTICES = [
  -1, -1,
   1, -1,
  -1,  1,
   1,  1,
];

export const COMMON_DEFS = `
struct Slice {
  offset: u32,
  length: u32,
}
`;

export function writeToBuffer(
  queue: GPUQueue,
  buffer: GPUBuffer,
  data: Uint8Array | Uint32Array | Float32Array
) {
  queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
}
