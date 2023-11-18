/**
 * Search for a string within a given binary
 *
 * @param bytes The total binary to search
 * @param string The string to be searched for, assumed to be in UTF-8
 * @param fromIndex The starting index to begin the search from (inclusive)
 * @returns The index of the found result, or -1 if no result is found
 */
export function findString(
  bytes: Uint8Array,
  string: string,
  fromIndex: number = 0
): number {
  const strBytes = new TextEncoder().encode(string);

  const maxSearchIndex = bytes.length - strBytes.length;

  let matchedIndex = -1;
  for (let i = fromIndex; matchedIndex === -1 && i < maxSearchIndex; i++) {
    let matched = true;
    for (let j = 0; matched && j < strBytes.length; j++) {
      matched = strBytes[j] === bytes[j + i];
    }

    if (matched) {
      matchedIndex = i;
    }
  }

  return matchedIndex;
}
