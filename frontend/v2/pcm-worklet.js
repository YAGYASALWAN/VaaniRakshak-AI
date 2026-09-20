class PCM16CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunkSize = 4096;
    this.buffer = new Int16Array(this.chunkSize);
    this.offset = 0;
    this.port.onmessage = event => {
      if (event.data?.type === 'flush') this.flush();
    };
  }

  flush() {
    if (!this.offset) return;
    const output = this.buffer.slice(0, this.offset);
    this.port.postMessage(output.buffer, [output.buffer]);
    this.buffer = new Int16Array(this.chunkSize);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!input) return true;
    for (let i = 0; i < input.length; i += 1) {
      const sample = Math.max(-1, Math.min(1, input[i]));
      this.buffer[this.offset++] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
      if (this.offset === this.chunkSize) this.flush();
    }
    return true;
  }
}

registerProcessor('pcm16-capture', PCM16CaptureProcessor);
