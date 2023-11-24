export const GosperGun = `#N Gosper glider gun
#C This was the first gun discovered.
#C As its name suggests, it was discovered by Bill Gosper.
x = 36, y = 9
24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4b
obo$10bo5bo7bo$11bo3bo$12b2o!`;

/**
 * Parses the run-length-encoding format used for a lot of game-of-life stuff.
 *
 * https://conwaylife.com/wiki/Run_Length_Encoded
 */
export function parseRLE(content: string) {
  const withoutComments = content.replace(/#[^\n]+\n/g, "");
  let [header] = withoutComments.split("\n", 1);

  const [, xStr, yStr] = /^x=(\d+),y=(\d+)$/.exec(header.replace(/\s+/g, ""))!;

  const width = parseInt(xStr, 10);
  const height = parseInt(yStr, 10);

  let body = withoutComments.substring(header.length + 1);
  body = body.replace(/\s+/g, "");

  const lines = body.split("!", 1)[0].split("$");

  const cells: (0 | 1)[] = [];
  for (let y = 0; y < height; y++) {
    let line = lines[y];

    const base = cells.length;
    const regex = /(\d*)(b|o)/g;

    let x = 0;
    let match;
    while ((match = regex.exec(line))) {
      const [, lengthStr, type] = match;
      const value = type === "o" ? 1 : 0;

      let length = lengthStr ? parseInt(lengthStr, 10) : 1;
      while (length--) {
        cells[base + x++] = value;
      }
    }

    while (x < width) {
      cells[base + x++] = 0;
    }
  }

  return { width, height, cells };
}
