class StreamingPlayerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffers = [];
    this.cursor = 0;

    this.port.onmessage = (event) => {
      const data = event.data;
      if (data instanceof Float32Array) {
        this.buffers.push(data);
      } else if (data.command === "clear") {
        this.buffers = [];
        this.cursor = 0;
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0][0];
    output.fill(0);

    let i = 0;
    while (i < output.length && this.buffers.length > 0) {
      const current = this.buffers[0];
      const available = current.length - this.cursor;
      const needed = output.length - i;
      const copySize = Math.min(available, needed);

      output.set(current.subarray(this.cursor, this.cursor + copySize), i);

      i += copySize;
      this.cursor += copySize;

      if (this.cursor >= current.length) {
        this.buffers.shift();
        this.cursor = 0;
      }
    }

    return true;
  }
}

registerProcessor("streaming-player", StreamingPlayerProcessor);
