/** Canonical, insertion-order AST serializer shared by snapshot and fuzz tools. */
export function emit(node, indent = '') {
  if (node == null) return `${indent}-\n`;
  if (Array.isArray(node)) return node.map(value => emit(value, indent)).join('');
  if (node instanceof Map) {
    let out = '';
    for (const [key, value] of node) out += `${indent}key\n${emit(key, `${indent}  `)}${indent}value\n${emit(value, `${indent}  `)}`;
    return out;
  }
  if (typeof node !== 'object') return `${indent}${JSON.stringify(node)}\n`;
  let out = `${indent}${node.type}\n`;
  for (const [key, value] of Object.entries(node)) {
    if (key !== 'type') out += `${indent}  ${key}\n${emit(value, `${indent}    `)}`;
  }
  return out;
}
