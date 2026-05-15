import assert from 'node:assert';
import { spawn } from 'node:child_process';

export class LspClient {
  constructor(command, args, options = {}) {
    this.child = spawn(command, args, {
      cwd: options.cwd,
      stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.nextId = 1;
    this.pending = new Map();
    this.notifications = [];
    this.buffer = Buffer.alloc(0);
    this.stderr = '';

    this.child.stdout.on('data', chunk => this._onData(chunk));
    this.child.stderr.on('data', chunk => { this.stderr += String(chunk); });
  }

  _onData(chunk) {
    this.buffer = Buffer.concat([this.buffer, chunk]);
    for (;;) {
      const headerEnd = this.buffer.indexOf('\r\n\r\n');
      if (headerEnd === -1) return;
      const header = this.buffer.slice(0, headerEnd).toString('ascii');
      const lengthMatch = header.match(/Content-Length:\s*(\d+)/i);
      assert.ok(lengthMatch, `missing Content-Length in ${header}`);
      const length = Number(lengthMatch[1]);
      const bodyStart = headerEnd + 4;
      const bodyEnd = bodyStart + length;
      if (this.buffer.length < bodyEnd) return;
      const body = this.buffer.slice(bodyStart, bodyEnd).toString('utf8');
      this.buffer = this.buffer.slice(bodyEnd);
      this._onMessage(JSON.parse(body));
    }
  }

  _onMessage(message) {
    if (Object.prototype.hasOwnProperty.call(message, 'id')) {
      const pending = this.pending.get(message.id);
      if (pending) {
        this.pending.delete(message.id);
        pending.resolve(message);
        return;
      }
    }
    this.notifications.push(message);
  }

  _write(message) {
    const body = JSON.stringify(message);
    this.child.stdin.write(`Content-Length: ${Buffer.byteLength(body, 'utf8')}\r\n\r\n${body}`);
  }

  request(method, params) {
    const id = this.nextId++;
    this._write({ jsonrpc: '2.0', id, method, params });
    return this._waitForResponse(id);
  }

  notify(method, params) {
    this._write({ jsonrpc: '2.0', method, params });
  }

  _waitForResponse(id) {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`timed out waiting for response ${id}; stderr: ${this.stderr}`));
      }, 3000);
      this.pending.set(id, {
        resolve: message => {
          clearTimeout(timeout);
          resolve(message);
        },
      });
    });
  }

  waitForNotification(method, predicate = () => true) {
    return new Promise((resolve, reject) => {
      const started = Date.now();
      const timer = setInterval(() => {
        const index = this.notifications.findIndex(message =>
          message.method === method && predicate(message.params || {}));
        if (index !== -1) {
          const [message] = this.notifications.splice(index, 1);
          clearInterval(timer);
          resolve(message);
          return;
        }
        if (Date.now() - started > 3000) {
          clearInterval(timer);
          reject(new Error(`timed out waiting for ${method}; stderr: ${this.stderr}`));
        }
      }, 10);
    });
  }

  async close() {
    if (!this.child.killed) {
      this.child.kill();
    }
  }
}
