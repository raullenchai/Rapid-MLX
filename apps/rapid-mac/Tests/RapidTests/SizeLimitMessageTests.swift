import Foundation
import Testing
@testable import Rapid

@Suite("SizeLimit messages use decimal file sizes")
struct SizeLimitMessageTests {
    @Test func attachmentAndUploadLimits() throws {
        let cases: [(String?, Int, String)] = [
            (AudioClientError.fileTooLarge(maxBytes: AudioClient.maxUploadBytes).errorDescription, AudioClient.maxUploadBytes, "25 MB"),
            (ChatFileAttachment.ValidationError.tooLarge.errorDescription, ChatFileAttachment.maxSourceBytes, "100 MB"),
            (ChatImageAttachment.ValidationError.tooLarge.errorDescription, ChatImageAttachment.maxBytes, "20 MB"),
            (ImportedEditImageError.tooLarge.errorDescription, ImageClient.maxEditImageBytes, "25 MB"),
        ]
        for (description, bytes, oldLabel) in cases {
            let message = try #require(description)
            #expect(message.contains(DownloadProgress.formatBytes(Int64(bytes))))
            #expect(!message.contains(oldLabel))
        }
    }

    @Test func documentValidationReportsActualLimit() throws {
        do {
            _ = try ChatFileAttachment(filename: "large.txt", kind: .txt, extractedText: "text", sourceByteCount: ChatFileAttachment.maxSourceBytes + 1)
            Issue.record("Oversized document was accepted")
        } catch let error as ChatFileAttachment.ValidationError {
            #expect(error == .tooLarge)
            #expect(error.errorDescription?.contains(DownloadProgress.formatBytes(Int64(ChatFileAttachment.maxSourceBytes))) == true)
            #expect(error.errorDescription?.contains("100 MB") == false)
        }
    }

    @Test func headerLimitReportsActualBytes() throws {
        let data = Data((String(repeating: "x", count: IPHTTPResponseParser.maxHeaderBytes + 1) + "\r\n\r\n").utf8)
        do {
            _ = try IPHTTPResponseParser.parse(data: data, url: URL(string: "https://example.com")!)
            Issue.record("Oversized headers were accepted")
        } catch {
            #expect(error.localizedDescription.contains(DownloadProgress.formatBytes(Int64(IPHTTPResponseParser.maxHeaderBytes))))
            #expect(!error.localizedDescription.contains("128 KB"))
        }
    }
}
