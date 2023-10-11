export type Result<T, Error = unknown> =
  | {
      success: true;
      value: T;
    }
  | {
      success: false;
      error: Error;
    };
