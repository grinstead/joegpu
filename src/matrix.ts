const tempA = new Float32Array(16);
const tempB = new Float32Array(16);

/**
 * A class that holds a 4x4 matrix that can be mutated into
 */
export class MutatingMatrix {
  constructor(public readonly data: Float32Array) {}

  /**
   * Changes the underlying matrix to the identity
   * @returns this matrix
   */
  reset() {
    const { data } = this;

    for (let row = 0, i = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++, i++) {
        data[i] = row === col ? 1 : 0;
      }
    }

    return this;
  }

  /**
   * Mutates the matrix to be `<multiplier> x <current data>`
   * @param multiplier the left hand side of the multiplication expression
   * @returns this matrix
   */
  apply(multiplier: Float32Array): this {
    const { data } = this;

    // write the multiplication result into the scratch tempA matrix
    for (let row = 0; row < 4; row++) {
      for (let col = 0; col < 4; col++) {
        let rowByCol = 0;
        for (let i = 0; i < 4; i++) {
          rowByCol += multiplier[4 * row + i] * data[col + 4 * i];
        }

        tempA[4 * row + col] = rowByCol;
      }
    }

    // write the result back into our data
    data.set(tempA);

    return this;
  }

  rotateAroundX(theta: number): this {
    const sin = Math.sin(theta);
    const cos = Math.cos(theta);

    // prettier-ignore
    tempB.set([
         1,    0,    0, 0,
         0,  cos, -sin, 0,
         0,  sin,  cos, 0,
         0,    0,    0, 1,
    ]);

    return this.apply(tempB);
  }

  rotateAroundY(theta: number): this {
    const sin = Math.sin(theta);
    const cos = Math.cos(theta);

    // prettier-ignore
    tempB.set([
       cos,    0,  sin, 0,
         0,    1,    0, 0,
      -sin,    0,  cos, 0,
         0,    0,    0, 1,
    ]);

    return this.apply(tempB);
  }

  rotateAroundZ(theta: number): this {
    const sin = Math.sin(theta);
    const cos = Math.cos(theta);

    // prettier-ignore
    tempB.set([
      cos, -sin,    0, 0,
      sin,  cos,    0, 0,
        0,    0,    1, 0,
        0,    0,    0, 1,
   ]);

    return this.apply(tempB);
  }

  translate(x: number, y: number, z: number) {
    // prettier-ignore
    tempB.set([
      1, 0, 0, x,
      0, 1, 0, y,
      0, 0, 1, z,
      0, 0, 0, 1,
    ]);

    return this.apply(tempB);
  }

  scale(x: number, y: number, z: number): this {
    // prettier-ignore
    tempB.set([
      x, 0, 0, 0,
      0, y, 0, 0,
      0, 0, z, 0,
      0, 0, 0, 1,
    ]);

    return this.apply(tempB);
  }

  transpose() {
    const { data } = this;

    for (let row = 0; row < 4; row++) {
      for (let col = row + 1; col < 4; col++) {
        const i = row * 4 + col;
        const j = col * 4 + row;
        const temp = data[i];
        data[i] = data[j];
        data[j] = temp;
      }
    }
  }
}
