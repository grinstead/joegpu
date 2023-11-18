export const NUM_BYTES_FLOAT32 = 4;

export type Result<T, Error = unknown> =
  | {
      success: true;
      value: T;
    }
  | {
      success: false;
      error: Error;
    };
