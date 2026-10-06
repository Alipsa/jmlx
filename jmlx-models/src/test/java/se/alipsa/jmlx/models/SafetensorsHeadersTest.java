package se.alipsa.jmlx.models;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SafetensorsHeadersTest {
  @TempDir Path temporaryDirectory;

  @Test
  void malformedHeaderJsonFailsWithIoExceptionNotUncheckedJackson() throws IOException {
    byte[] body = "{not json".getBytes(StandardCharsets.UTF_8);
    ByteBuffer file = ByteBuffer.allocate(Long.BYTES + body.length).order(ByteOrder.LITTLE_ENDIAN);
    file.putLong(body.length);
    file.put(body);
    Path corrupt = temporaryDirectory.resolve("corrupt.safetensors");
    Files.write(corrupt, file.array());
    IOException error =
        assertThrows(IOException.class, () -> SafetensorsHeaders.tensorNames(List.of(corrupt)));
    assertTrue(error.getMessage().contains("corrupt.safetensors"));
    assertTrue(error.getCause() instanceof tools.jackson.core.JacksonException);
  }

  @Test
  void readFullyAcceptsPartialChannelReads() throws IOException {
    ByteBuffer target = ByteBuffer.allocate(Long.BYTES);
    SafetensorsHeaders.readFully(
        new ChunkedChannel(new byte[] {1, 2, 3, 4, 5, 6, 7, 8}), target, "EOF");
    assertArrayEquals(new byte[] {1, 2, 3, 4, 5, 6, 7, 8}, target.array());
  }

  @Test
  void readFullyRejectsActualEof() {
    var error =
        assertThrows(
            IllegalArgumentException.class,
            () ->
                SafetensorsHeaders.readFully(
                    new ChunkedChannel(new byte[] {1, 2, 3}),
                    ByteBuffer.allocate(Long.BYTES),
                    "EOF"));
    assertTrue(error.getMessage().contains("EOF"));
  }

  private static final class ChunkedChannel implements ReadableByteChannel {
    private final byte[] source;
    private int offset;
    private boolean open = true;

    private ChunkedChannel(byte[] source) {
      this.source = source;
    }

    @Override
    public int read(ByteBuffer target) {
      if (offset == source.length) {
        return -1;
      }
      int size = Math.min(2, Math.min(source.length - offset, target.remaining()));
      target.put(source, offset, size);
      offset += size;
      return size;
    }

    @Override
    public boolean isOpen() {
      return open;
    }

    @Override
    public void close() {
      open = false;
    }
  }
}
