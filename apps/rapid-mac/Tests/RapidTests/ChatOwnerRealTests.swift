import Darwin
import Foundation
import Testing
@testable import Rapid

private func realOwnerPythonAvailable() -> Bool {
    guard let path = ProcessInfo.processInfo.environment["RAPID_MLX_CHAT_OWNER_PYTHON"],
          FileManager.default.isExecutableFile(atPath: path) else { return false }
    let process = Process()
    process.executableURL = URL(fileURLWithPath: path)
    process.arguments = ["-c", "import rapid_mlx.chat_runtime.owner"]
    process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
    let finished = DispatchSemaphore(value: 0)
    process.terminationHandler = { _ in finished.signal() }
    do { try process.run() } catch { return false }
    guard finished.wait(timeout: .now() + 10) == .success else {
        process.terminate()
        return false
    }
    return process.terminationStatus == 0
}

private actor RealProjection {
    var reducer: ChatTurnReducer
    init(_ id: String) { reducer = ChatTurnReducer(turnID: id) }
    func apply(_ event: ChatWireEnvelope) throws { _ = try reducer.apply(event) }
    func snapshot() -> ChatTurnProjection { reducer.projection }
}

@Suite(.serialized) struct ChatOwnerRealTests {
    @Test(.enabled(if: realOwnerPythonAvailable()))
    @MainActor func realOwnerLifecycle() async throws {
        let python = try #require(ProcessInfo.processInfo.environment["RAPID_MLX_CHAT_OWNER_PYTHON"])
        let root = URL(fileURLWithPath: "/private/tmp/rapid-mlx-chat-cli-slice/p04-scratch/e2e-home-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = root.appendingPathComponent("workspace")
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        try "fixture read output".write(to: workspace.appendingPathComponent("sample.txt"), atomically: true, encoding: .utf8)
        let fake = Process()
        fake.executableURL = URL(fileURLWithPath: python)
        fake.arguments = ["-u", "-c", #"""
import json, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class Handler(BaseHTTPRequestHandler):
 def log_message(self, *args): pass
 def do_POST(self):
  req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
  messages = req.get('messages', [])
  user = next((m.get('content','') for m in reversed(messages) if m.get('role') == 'user'), '')
  if isinstance(user, list): user = ''.join(p.get('text','') for p in user if isinstance(p, dict))
  user = user.split('\n')[-1]
  done = bool(messages and messages[-1].get('role') == 'tool')
  self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
  if user == 'stop':
   time.sleep(15)
   return
  if user in ('read', 'allow', 'deny') and not done:
   name = 'read_file' if user == 'read' else 'write_file'
   args = {'path':'sample.txt'} if user == 'read' else {'path':user+'.txt','content':'written'}
   delta = {'tool_calls':[{'index':0,'id':'call_'+user,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
   chunk = {'id':'fixture','object':'chat.completion.chunk','created':1,'model':'fixture','choices':[{'index':0,'delta':delta,'finish_reason':None}]}
   self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode())
   chunk['choices']=[{'index':0,'delta':{},'finish_reason':'tool_calls'}]
   self.wfile.write(('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode()); self.wfile.flush()
   return
  chunk = {'id':'fixture','object':'chat.completion.chunk','created':1,'model':'fixture','choices':[{'index':0,'delta':{'content':'Hello from fixture.'},'finish_reason':None}]}
  self.wfile.write(('data: '+json.dumps(chunk)+'\n\n').encode())
  chunk['choices']=[{'index':0,'delta':{},'finish_reason':'stop'}]
  self.wfile.write(('data: '+json.dumps(chunk)+'\n\ndata: [DONE]\n\n').encode()); self.wfile.flush()
s = ThreadingHTTPServer(('127.0.0.1',0), Handler)
print(s.server_port, flush=True)
s.serve_forever()
"""#]
        let pipe = Pipe(); fake.standardOutput = pipe; fake.standardError = FileHandle.nullDevice
        try fake.run()
        defer { if fake.isRunning { fake.terminate() } }
        var portBytes = Data()
        while let byte = try pipe.fileHandleForReading.read(upToCount: 1), !byte.isEmpty, byte != Data([10]) { portBytes.append(byte) }
        let port = try #require(Int(String(decoding: portBytes, as: UTF8.self)))
        var env = ProcessInfo.processInfo.environment
        env["RAPID_MLX_CHAT_HOME"] = root.path; env["RAPID_MLX_CHAT_IDLE_SECONDS"] = "2"
        let runtime = ChatOwnerRuntime(interpreter: URL(fileURLWithPath: python), baseEnvironment: env)
        let a = try ChatOwnerConnection(runtime: runtime, environment: env)
        let b = try ChatOwnerConnection(runtime: runtime, environment: env)
        async let first = a.status()
        async let second = b.status()
        let (owner, other) = try await (first, second)
        #expect(owner.pid == other.pid && owner.ownerEpoch == other.ownerEpoch)
        // The unique, initially empty home establishes that this test spawned this owner.
        func stop(_ status: ChatOwnerStatus) {
            if let start = ChatOwnerDiscovery.liveProcessStartTime(status.pid), abs(start - status.processStartTime) <= 0.01 { _ = kill(status.pid, SIGTERM) }
        }
        defer { stop(owner) }
        let config = ChatBackendConfig(defaults: UserDefaults(suiteName: UUID().uuidString)!)
        config.chatBackend = .ownerV1
        let model = ChatViewModel(client: ChatStreamClient(baseURL: URL(string: "http://127.0.0.1:\(port)/v1")!), chatBackendConfig: config, ownerClientFactory: { try await a.turnClient() }, ownerWorkspace: workspace, persistsConversations: false)
        model.send("hello", alias: "fixture")
        await model._testingWaitForCurrentTurn()
        #expect(model.messages.contains { $0.role == .assistant && $0.content.contains("Hello from fixture.") })
        #expect(model.conversations.first?.backend == .ownerV1)
        let storedID = try #require(model.conversations.first?.ownerConversationID)
        let stored = try await a.turnClient().conversation(id: storedID)
        if case .object(let fields) = stored {
            if case .string(let path) = fields["workspace"] {
                #expect(URL(fileURLWithPath: path).resolvingSymlinksInPath().standardizedFileURL.path == workspace.resolvingSymlinksInPath().standardizedFileURL.path)
            } else { Issue.record("Missing workspace") }
        } else { Issue.record("Missing owner conversation") }
        let bearerURL = root.appendingPathComponent("default/run/" + owner.credentialFile)
        let token = try String(contentsOf: bearerURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        model.send("read", alias: "fixture")
        await model._testingWaitForCurrentTurn()
        let readRows = model.messages.filter { $0.role == .tool && $0.content.contains("fixture read output") }
        #expect(readRows.count == 1)
        let readProjection = try #require(await model._testingOwnerProjection())
        let readReplay = RealProjection(readProjection.turnID)
        var replayRequest = URLRequest(url: owner.endpoint.appendingPathComponent("v1/rapid/chat/turns/" + readProjection.turnID + "/events"))
        replayRequest.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
        replayRequest.setValue("\(readProjection.turnID):0", forHTTPHeaderField: "Last-Event-ID")
        let (replayData, replayResponse) = try await URLSession.shared.data(for: replayRequest)
        #expect((replayResponse as? HTTPURLResponse)?.statusCode == 200)
        var replayDecoder = ChatSSEDecoder()
        for event in try replayDecoder.append(replayData) { try await readReplay.apply(event) }
        try replayDecoder.finish()
        #expect(await readReplay.snapshot() == readProjection)
        for (text, allow) in [("allow", true), ("deny", false)] {
            model.send(text, alias: "fixture")
            for _ in 0..<500 where model.ownerApprovals.isEmpty { try await Task.sleep(for: .milliseconds(20)) }
            let approval = try #require(model.ownerApprovals.values.first)
            model.resolveOwnerPermission(opID: approval.opID, allow: allow)
            await model._testingWaitForCurrentTurn()
            #expect(FileManager.default.fileExists(atPath: workspace.appendingPathComponent(text + ".txt").path) == allow)
        }
        model.send("stop", alias: "fixture")
        try await Task.sleep(for: .milliseconds(500))
        model.stop()
        for _ in 0..<500 {
            if !model.isStreaming, await model._testingOwnerProjection()?.terminal != nil { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        #expect(!model.isStreaming)
        #expect(await model._testingOwnerProjection()?.terminal?.outcome == .cancelled)
        func request(_ status: ChatOwnerStatus, _ token: String, _ path: String, method: String, body: String? = nil) async throws -> Data {
            var req = URLRequest(url: status.endpoint.appendingPathComponent("v1/rapid/chat/" + path))
            req.httpMethod = method; req.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
            if let body { req.httpBody = Data(body.utf8); req.setValue("application/json", forHTTPHeaderField: "Content-Type") }
            let (data, response) = try await URLSession.shared.data(for: req)
            let code = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard (200..<300).contains(code) else { throw ChatOwnerConnectionError.requestFailed(code) }
            return data
        }
        _ = try await request(owner, token, "clients", method: "POST", body: #"{"client_id":"raw-client"}"#)
        let data = try await request(owner, token, "conversations", method: "POST", body: "{}")
        let conversation = try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        let id = try #require(conversation["id"] as? String)
        let branch = try #require(conversation["branch_id"] as? String)
        let turn = UUID().uuidString.lowercased()
        let intent = ChatTurnRequest(turnID: turn, conversationID: id, branchID: branch, expectedRevision: 0, text: "hello", inferenceURL: "http://127.0.0.1:\(port)/v1", model: "fixture", inferenceAPIKey: "test")
        let projection = RealProjection(turn)
        let client = try await a.turnClient()
        try await client.submit(intent, after: 0) { try await projection.apply($0) }
        let result = await projection.snapshot()
        #expect(result.terminal?.outcome == .completed)
        #expect(result.parts.contains { if case .text(_, let text) = $0 { return text.contains("Hello from fixture.") }; return false })
        let raw = try URLSessionChatTurnClient(endpoint: owner.endpoint, bearer: token)
        let replay = RealProjection(turn)
        try await raw.stream(turnID: turn, after: 0) { try await replay.apply($0) }
        #expect(await replay.snapshot() == result)
        stop(owner)
        for _ in 0..<100 where ChatOwnerDiscovery.liveProcessStartTime(owner.pid) != nil { try await Task.sleep(for: .milliseconds(50)) }
        _ = try await a.reconnect()
        let replacement = try await a.status()
        defer { stop(replacement) }
        let newToken = try String(contentsOf: root.appendingPathComponent("default/run/" + replacement.credentialFile), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        #expect(replacement.ownerEpoch != owner.ownerEpoch && newToken != token)
        await a.close(); await b.close()
        for _ in 0..<120 where ChatOwnerDiscovery.liveProcessStartTime(replacement.pid) != nil { try await Task.sleep(for: .milliseconds(50)) }
        #expect(ChatOwnerDiscovery.liveProcessStartTime(replacement.pid) == nil)
    }
}
