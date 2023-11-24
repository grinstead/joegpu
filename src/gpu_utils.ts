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

/**
 * All uniform vriables must be passed in on 16-byte alignment
 */
export const UNIFORM_ALIGNMENT = 16;

export function writeToBuffer(
  queue: GPUQueue,
  buffer: GPUBuffer,
  data: Uint8Array | Uint32Array | Float32Array
) {
  queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
}
