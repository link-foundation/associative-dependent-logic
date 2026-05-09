function utf8Encode(text) {
  if (typeof TextEncoder !== 'undefined') return new TextEncoder().encode(text);
  return Uint8Array.from(Buffer.from(text, 'utf8'));
}

function utf8Decode(bytes) {
  if (typeof TextDecoder !== 'undefined') return new TextDecoder().decode(bytes);
  return Buffer.from(bytes).toString('utf8');
}

function bytesToBase64(bytes) {
  if (typeof btoa !== 'undefined') {
    let binary = '';
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary);
  }
  return Buffer.from(bytes).toString('base64');
}

function base64ToBytes(base64) {
  if (typeof atob !== 'undefined') {
    return Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  }
  return Uint8Array.from(Buffer.from(base64, 'base64'));
}

function toBase64Url(base64) {
  return base64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '');
}

function fromBase64Url(base64url) {
  const base64 = base64url.replace(/-/g, '+').replace(/_/g, '/');
  return base64.padEnd(Math.ceil(base64.length / 4) * 4, '=');
}

export function normalizePlaygroundState(state) {
  return {
    example: typeof state?.example === 'string' ? state.example : '',
    source: typeof state?.source === 'string' ? state.source : '',
  };
}

export function encodePlaygroundState(state) {
  const normalized = normalizePlaygroundState(state);
  const json = JSON.stringify(normalized);
  return `#state=${toBase64Url(bytesToBase64(utf8Encode(json)))}`;
}

export function decodePlaygroundState(hash) {
  const rawHash = typeof hash === 'string' ? hash.replace(/^#/, '') : '';
  const params = new URLSearchParams(rawHash);
  const encoded = params.get('state');
  if (!encoded) return null;
  try {
    const bytes = base64ToBytes(fromBase64Url(encoded));
    return normalizePlaygroundState(JSON.parse(utf8Decode(bytes)));
  } catch (_) {
    return null;
  }
}
