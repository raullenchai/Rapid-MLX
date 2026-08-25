import Darwin
import Foundation
import Testing
@testable import Rapid

/// Regression coverage for the crash-safe pipe read helpers
/// (`FileHandleSafeRead.swift`).
///
/// - `readSafely` / `readToEndSafely` replace the `availableData` /
///   `readDataToEndOfFile()` family, which raises an uncatchable
///   `NSFileHandleOperationException` on a bad descriptor (SIGABRT).
/// - `PipeDrainer` is the non-blocking, crash-safe drainer used by every
///   subprocess-pipe readabilityHandler AND the SidecarExtractor
///   terminationHandler tail. It must: read buffered bytes, never block on
///   a write end still held open, never crash on a bad FD, report genuine
///   EOF distinctly from "nothing right now", and — because it sets
///   `O_NONBLOCK` once and never toggles per-drain — stay safe under
///   concurrent drains of the same pipe (the codex r2 BLOCKING).
@Suite("Crash-safe FileHandle reads")
struct FileHandleSafeReadTests {

    /// `PipeDrainer` returns whatever is already buffered, not at EOF while
    /// the write end stays open.
    @Test("PipeDrainer drains already-buffered bytes")
    func drainReturnsBufferedBytes() throws {
        let pipe = Pipe()
        let drainer = PipeDrainer(pipe.fileHandleForReading)   // live handle
        let payload = Data("tail".utf8)
        try pipe.fileHandleForWriting.write(contentsOf: payload)
        let result = drainer.drain()
        #expect(result.data == payload)
        #expect(result.atEOF == false)          // writer still open
        try? pipe.fileHandleForWriting.close()
        try? pipe.fileHandleForReading.close()
    }

    /// The critical no-hang contract: with the write end held OPEN and
    /// nothing buffered — a descendant still holding stderr — a blocking
    /// read would stall forever. `PipeDrainer` must return empty at once
    /// (not EOF). The 1-minute limit turns a regression-to-blocking into a
    /// clean failure instead of wedging the whole suite.
    @Test(
        "PipeDrainer returns empty (never blocks) while the write end stays open",
        .timeLimit(.minutes(1))
    )
    func drainDoesNotHangWithOpenWriteEnd() {
        let pipe = Pipe()
        let drainer = PipeDrainer(pipe.fileHandleForReading)
        let writer = pipe.fileHandleForWriting  // stays open, nothing written
        let reader = pipe.fileHandleForReading
        defer {
            try? writer.close()
            try? reader.close()
        }
        let result = drainer.drain()
        #expect(result.data.isEmpty)
        #expect(result.atEOF == false)          // EAGAIN, not EOF
    }

    /// A closed write end signals EOF: the drain reports `atEOF` so a
    /// handler can detach. (Detaching on empty-only would misfire on the
    /// no-data case above.)
    @Test("PipeDrainer reports atEOF once every writer closes")
    func drainReportsEOFOnClosedWriter() throws {
        let pipe = Pipe()
        let drainer = PipeDrainer(pipe.fileHandleForReading)
        try pipe.fileHandleForWriting.close()   // no more writers → EOF
        let result = drainer.drain()
        #expect(result.data.isEmpty)
        #expect(result.atEOF == true)
        try? pipe.fileHandleForReading.close()
    }

    /// A descriptor closed UNDERNEATH the drainer (the fd-reuse race the
    /// owning-handle design guards) must degrade to empty rather than
    /// crashing — raw `read(2)` surfaces EBADF as -1, never an NSException.
    ///
    /// The drainer is constructed while its descriptor is LIVE (so `ready` is
    /// true and `drain()` really issues a `read(2)` on the now-bad fd — a
    /// regression that crashes on `EBADF` or reads a recycled descriptor is
    /// caught). Deterministic-guard (#2318): that descriptor is a private HIGH
    /// number (via `F_DUPFD`, first free fd >= 1024), closed only AFTER the
    /// drainer initializes, and never handed back to the low-fd allocator a
    /// concurrent test's `Pipe()` uses. So `read(high)` always hits a
    /// genuinely-`EBADF` descriptor and can never consume another test's
    /// bytes. The LOW fd is released via Foundation's `FileHandle.close()` so
    /// its deinit cannot later re-`close(2)` a number already recycled. (The
    /// pre-fix form drained the recycled LOW read fd directly — the OS handed
    /// that number to a concurrent test's pipe, so the bad drain read that
    /// test's buffered bytes while the buffered drain read nothing — the #2318
    /// paired failure.)
    @Test("PipeDrainer returns empty on a bad descriptor instead of crashing")
    func drainOnBadDescriptorDoesNotCrash() throws {
        let pipe = Pipe()
        let low = pipe.fileHandleForReading.fileDescriptor
        // Pin a private HIGH duplicate of the read end. Concurrent test Pipes
        // only ever take the lowest free fds, so `high` is never routed to them
        // once freed. Derive the minimum from the real soft RLIMIT_NOFILE so the
        // F_DUPFD target is always in-range (avoid assuming 1024).
        var rl = rlimit()
        getrlimit(RLIMIT_NOFILE, &rl)
        let soft = rl.rlim_cur
        let highMin: Int32 = soft > 1024 ? 1024 : max(8, Int32(soft) - 8)
        let high = fcntl(low, F_DUPFD, highMin)
        // Abort cleanly (skip) rather than building a FileHandle over -1 if the
        // environment can't give us a private high descriptor.
        try #require(high >= 0, "F_DUPFD failed to pin a private high bad fd")
        // Guarantee the raw `high` fd is closed on EVERY exit path (incl. a throw
        // from FileHandle.close() below). `highClosed` tracks whether the explicit
        // `close(high)` below has already run — the deferred close must NOT fire on a
        // number the OS may have reallocated in between, or it would close an unrelated
        // live descriptor (the exact corruption class this test pins against).
        var highClosed = false
        defer { if !highClosed { close(high) } }
        // Construct while `high` is live → ready=true (fd captured, O_NONBLOCK).
        let drainer = PipeDrainer(FileHandle(fileDescriptor: high, closeOnDealloc: false))
        // Tear the READ side down from underneath the live drainer, but leave the
        // WRITE end open for the duration of the drain: a still-valid descriptor
        // would then read EAGAIN → atEOF == false, so `atEOF == true` below proves
        // the EBADF path was actually exercised (not ordinary EOF).
        try pipe.fileHandleForReading.close()    // release low via Foundation (no double-close)
        close(high)                              // free the pinned high — now bad, and private
        highClosed = true                        // suppress the deferred close: fd already freed
        // Prove the descriptor really is EBADF at this instant (fcntl F_GETFD on a
        // closed fd → -1/EBADF) in the same synchronous sequence as the drain, so a
        // regression like an ineffective close() is caught. Snapshot BOTH the return
        // value and errno immediately after fcntl — Swift Testing's #expect machinery
        // may itself clobber the thread-local errno before we can read it. This test is
        // the ONLY one that ever dup()s into the >= highMin range, so `high` cannot
        // have been re-issued to a concurrent test's low-numbered Pipe().
        let fdState = fcntl(high, F_GETFD)
        let fdErrno = errno
        #expect(fdState == -1)
        #expect(fdErrno == EBADF)
        let result = drainer.drain()
        try? pipe.fileHandleForWriting.close()   // writer held open through the drain
        #expect(result.data.isEmpty)            // no crash, no bytes
        #expect(result.atEOF == true)           // only EBADF (writer still open) yields EOF
    }

    /// The codex r2 BLOCKING regression: two drains of the SAME pipe running
    /// concurrently, write end held open. With permanent `O_NONBLOCK` (no
    /// per-drain toggle) neither can strand the other in a blocking read;
    /// together they consume exactly the buffered bytes and both return
    /// promptly. The time limit fails cleanly if a regression re-introduces
    /// the toggle race and one drain wedges.
    @Test(
        "concurrent drains of one pipe never block and split the bytes exactly",
        .timeLimit(.minutes(1))
    )
    func concurrentDrainsDoNotBlockOrRace() async throws {
        let pipe = Pipe()
        let readerFD = pipe.fileHandleForReading.fileDescriptor
        let drainer = PipeDrainer(pipe.fileHandleForReading)
        let writer = pipe.fileHandleForWriting  // stays OPEN → no EOF

        // Direct, order-independent guard against the old per-drain toggle:
        // construction must set O_NONBLOCK once and leave it set. This holds
        // even if the two detached drains below happen to run serially (so
        // the toggle bug couldn't be masked by lack of true overlap).
        func isNonBlocking() -> Bool {
            let flags = fcntl(readerFD, F_GETFL)
            return flags != -1 && (flags & O_NONBLOCK) != 0
        }
        #expect(isNonBlocking())                 // set at construction

        let payload = Data(repeating: 0x41, count: 8 * 1024)
        try writer.write(contentsOf: payload)

        // Two concurrent drains share the one drainer.
        async let a = Task.detached { drainer.drain().data }.value
        async let b = Task.detached { drainer.drain().data }.value
        let total = await a.count + b.count
        // Every buffered byte is read exactly once across the two drains
        // (raw read consumes disjoint slices); no byte is lost or doubled.
        #expect(total == payload.count)
        // …and still non-blocking afterwards — a drain never restores
        // blocking mode underneath a concurrent one (codex r2 BLOCKING).
        #expect(isNonBlocking())

        try? writer.close()
        try? pipe.fileHandleForReading.close()
    }

    /// `readSafely` fills-to-count/EOF (see its doc comment): it blocks
    /// until `count` bytes arrive OR the write end signals EOF. With the
    /// writer closed FIRST, the pipe holds exactly `payload` then reports
    /// EOF, so the read returns the buffered bytes instead of blocking
    /// forever. The 1-minute limit turns a regression-to-unbounded-block
    /// (e.g. someone dropping the pre-close) into a clean failure rather
    /// than a wedged suite.
    @Test(
        "readSafely returns buffered bytes once the writer closes (EOF)",
        .timeLimit(.minutes(1))
    )
    func readSafelyReturnsBufferedBytes() throws {
        let pipe = Pipe()
        let payload = Data("hello".utf8)
        try pipe.fileHandleForWriting.write(contentsOf: payload)
        try pipe.fileHandleForWriting.close()   // EOF after the 5 bytes
        #expect(pipe.fileHandleForReading.readSafely(upToCount: safePipeChunkBytes) == payload)
        try? pipe.fileHandleForReading.close()
    }

    /// A closed descriptor must degrade to empty `Data` rather than
    /// raising the uncatchable `NSFileHandleOperationException`.
    @Test("readSafely + readToEndSafely return empty on a closed descriptor instead of crashing")
    func safeReadsOnClosedDescriptorDoNotCrash() throws {
        let pipe = Pipe()
        let reader = pipe.fileHandleForReading
        try reader.close()
        #expect(reader.readSafely(upToCount: safePipeChunkBytes).isEmpty)
        #expect(reader.readToEndSafely().isEmpty)
    }
}
