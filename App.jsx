import { API } from './api.js'
import { useState, useEffect, useRef, useCallback } from "react";

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────
const HISTORY_LIMIT = 10;

const COLORS = {
  greenDark: "#1a4a2e",
  greenMid:  "#2d6a45",
  gold:      "#c9a84c",
  cream:     "#f5f0e8",
  text:      "#2c2c2c",
  red:       "#eb5757",
  green:     "#6fcf97",
};

// ─────────────────────────────────────────────────────────────
// Simli WebRTC helpers (pure JS, live outside React)
// ─────────────────────────────────────────────────────────────
let ws             = null;
let pc             = null;
let silenceInterval = null;

function makeSilence(bytes = 6000) {
  return new Uint8Array(bytes);
}

function stopSilence() {
  if (silenceInterval) { clearInterval(silenceInterval); silenceInterval = null; }
}

function startSilenceKeepAlive() {
  stopSilence();
  silenceInterval = setInterval(() => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(makeSilence());
    } else {
      stopSilence();
    }
  }, 150);
}

function sendAudioToSimli(pcmArrayBuffer) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.warn("WebSocket not open — audio not sent");
    return;
  }
  stopSilence();
  const CHUNK = 6000;
  const bytes = new Uint8Array(pcmArrayBuffer);
  let sent = 0;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    ws.send(bytes.slice(i, i + CHUNK));
    sent++;
  }
  const durationMs = (bytes.length / 2 / 16000) * 1000;
  console.log(`Sent ${bytes.length} PCM bytes (${sent} chunks, ~${(durationMs / 1000).toFixed(1)}s)`);
  setTimeout(() => {
    if (ws && ws.readyState === WebSocket.OPEN) startSilenceKeepAlive();
  }, durationMs + 500);
}

function waitForIce(peerConn) {
  return new Promise((resolve) => {
    if (peerConn.iceGatheringState === "complete") return resolve();
    let stable = -1, cur = 0;
    function tick() {
      if (peerConn.iceGatheringState === "complete" || cur === stable) return resolve();
      stable = cur;
      setTimeout(tick, 250);
    }
    peerConn.onicecandidate = (e) => { if (!e.candidate) resolve(); else cur++; };
    setTimeout(tick, 500);
    setTimeout(resolve, 6000);
  });
}

function closeSimli() {
  if (ws) { try { ws.close(); } catch (_) {} ws = null; }
  if (pc) { try { pc.close(); } catch (_) {} pc = null; }
}

function playPCMLocally(pcmBuf) {
  try {
    const ctx  = new AudioContext({ sampleRate: 16000 });
    const view = new DataView(pcmBuf);
    const abuf = ctx.createBuffer(1, pcmBuf.byteLength / 2, 16000);
    const ch   = abuf.getChannelData(0);
    for (let i = 0; i < ch.length; i++) ch[i] = view.getInt16(i * 2, true) / 32768;
    const src  = ctx.createBufferSource();
    src.buffer = abuf;
    src.connect(ctx.destination);
    src.start();
  } catch (e) { console.error("Local audio error:", e); }
}

// ─────────────────────────────────────────────────────────────
// Sub-components
// ─────────────────────────────────────────────────────────────

function MicIcon({ color = "#555" }) {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2">
      <rect x="9" y="2" width="6" height="12" rx="3" />
      <path d="M19 10a7 7 0 01-14 0M12 19v3M8 22h8" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function AvatarPlaceholder() {
  return (
    <div style={{
      width: 140, height: 140, borderRadius: "50%",
      background: COLORS.greenDark,
      display: "flex", alignItems: "center", justifyContent: "center",
    }}>
      <svg width="90" height="90" viewBox="0 0 48 48" fill="none" style={{ opacity: 0.5 }}>
        <circle cx="24" cy="18" r="10" fill="white" />
        <path d="M4 42c0-11 8-18 20-18s20 7 20 18" fill="white" />
      </svg>
    </div>
  );
}

function ChatMessage({ msg }) {
  const isUser  = msg.role === "user";
  const isError = msg.role === "error";

  const bubbleStyle = {
    maxWidth: "70%",
    padding: "12px 16px",
    borderRadius: 12,
    fontSize: 14,
    lineHeight: 1.5,
    alignSelf: isUser ? "flex-end" : "flex-start",
    background: isUser
      ? COLORS.greenDark
      : isError
      ? "#fff3cd"
      : "white",
    color: isUser ? "white" : isError ? "#856404" : COLORS.text,
    border: isUser ? "none" : isError ? "1px solid #ffc107" : "1px solid #e0e0e0",
    borderBottomRightRadius: isUser ? 4 : 12,
    borderBottomLeftRadius:  isUser ? 12 : 4,
  };

  const lines = msg.text.split("\n");

  return (
    <div style={bubbleStyle}>
      <span>
        {lines.map((line, i) => (
          <span key={i}>
            {line}
            {i < lines.length - 1 && <br />}
          </span>
        ))}
      </span>
      <div style={{ fontSize: 10, color: isUser ? "rgba(255,255,255,0.6)" : "#aaa", marginTop: 4 }}>
        {msg.time}
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────
// Main App
// ─────────────────────────────────────────────────────────────
export default function App() {
  const [messages,       setMessages]       = useState([]);
  const [history,        setHistory]        = useState([]);
  const [inputText,      setInputText]      = useState("");
  const [lang,           setLang]           = useState("en");
  const [isSending,      setIsSending]      = useState(false);
  const [isRecording,    setIsRecording]    = useState(false);
  const [simliToken,     setSimliToken]     = useState(null);
  const [simliConnected, setSimliConnected] = useState(false);
  const [simliStatus,    setSimliStatus]    = useState({ text: "CONNECTING…", state: "" });
  const [videoVisible,   setVideoVisible]   = useState(false);

  const videoRef       = useRef(null);
  const audioRef       = useRef(null);
  const messagesEndRef = useRef(null);
  const mediaRecRef    = useRef(null);
  const audioChunks    = useRef([]);
  const simliTokenRef  = useRef(null);

  useEffect(() => { simliTokenRef.current = simliToken; }, [simliToken]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  function now() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function addMessage(text, role) {
    setMessages(prev => [...prev, { text, role, time: now() }]);
  }

  function setStatus(text, state) {
    setSimliStatus({ text, state });
  }

  // ── Simli ─────────────────────────────────────────────────
  const refreshSimliToken = useCallback(async () => {
    try {
      const res  = await fetch(API.startSession, { method: "POST" }); // ← API
      const data = await res.json();
      if (res.ok && data.simli_session_token) {
        setSimliToken(data.simli_session_token);
        console.log("Token refreshed ✓");
      }
    } catch (e) { console.warn("Token refresh failed:", e); }
  }, []);

  const connectSimli = useCallback(async (token) => {
    closeSimli();

    pc = new RTCPeerConnection({
      sdpSemantics: "unified-plan",
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    pc.addEventListener("track", (evt) => {
      console.log("Track received:", evt.track.kind);
      if (evt.track.kind === "video") {
        if (videoRef.current) videoRef.current.srcObject = evt.streams[0];
        setVideoVisible(true);
        setStatus("CONNECTED", "connected");
        setSimliConnected(true);
      } else {
        if (audioRef.current) audioRef.current.srcObject = evt.streams[0];
      }
    });

    pc.addEventListener("iceconnectionstatechange", () => {
      console.log("ICE:", pc.iceConnectionState);
      if (["failed", "disconnected", "closed"].includes(pc.iceConnectionState)) {
        setStatus("ERROR — RETRY", "error");
        setSimliConnected(false);
      }
    });

    pc.addTransceiver("video", { direction: "recvonly" });
    pc.addTransceiver("audio", { direction: "recvonly" });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitForIce(pc);
    const localDesc = pc.localDescription;

    const wsUrl = `wss://api.simli.ai/compose/webrtc/p2p?session_token=${encodeURIComponent(token)}&enableSFU=true`;
    ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("WebSocket open timeout")), 10000);

      ws.onopen = () => {
        clearTimeout(timeout);
        console.log("WebSocket open — sending offer…");
        ws.send(JSON.stringify({ type: localDesc.type, sdp: localDesc.sdp }));
      };

      ws.onmessage = async (evt) => {
        if (typeof evt.data === "string") {
          let msg;
          try { msg = JSON.parse(evt.data); } catch (_) {
            const raw = evt.data.trim();
            console.log("WS signal:", raw);
            if (raw === "STOP") refreshSimliToken();
            return;
          }
          if (msg.type === "answer") {
            console.log("Got RTC answer ✓");
            await pc.setRemoteDescription(msg);
            resolve();
          }
        }
      };

      ws.onerror  = () => { clearTimeout(timeout); reject(new Error("WebSocket error")); };
      ws.onclose  = (e) => {
        console.log("WS closed:", e.code, e.reason);
        ws = null;
        setSimliConnected(false);
        setStatus("READY", "connected");
      };
    });

    console.log("Simli WebRTC handshake complete ✓");
    startSilenceKeepAlive();
  }, [refreshSimliToken]);

  const initSimli = useCallback(async () => {
    setStatus("CONNECTING…", "");
    try {
      const res  = await fetch(API.startSession, { method: "POST" }); // ← API
      const data = await res.json();
      if (!res.ok || !data.simli_session_token) throw new Error(data.error || "No token");
      setSimliToken(data.simli_session_token);
      console.log("Got Simli token ✓");
      await connectSimli(data.simli_session_token);
    } catch (e) {
      console.error("Simli init error:", e);
      setStatus("ERROR — RETRY", "error");
    }
  }, [connectSimli]);

  useEffect(() => {
    initSimli();
    return () => { closeSimli(); stopSilence(); };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // ── Send message ───────────────────────────────────────────
  const sendMessage = useCallback(async (textOverride) => {
    if (isSending) return;
    const text = (textOverride ?? inputText).trim();
    if (!text) return;

    setIsSending(true);
    setInputText("");
    addMessage(text, "user");

    try {
      const res  = await fetch(API.chat, {           // ← API
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          history,
          simli_session_token: simliTokenRef.current,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Server error");

      if (data.simli_session_token) setSimliToken(data.simli_session_token);
      addMessage(data.answer, "bot");

      setHistory(prev => {
        const next = [...prev, [text, data.answer]];
        return next.length > HISTORY_LIMIT ? next.slice(-HISTORY_LIMIT) : next;
      });

      if (data.audio_b64) {
        const raw = atob(data.audio_b64);
        const buf = new ArrayBuffer(raw.length);
        const u8  = new Uint8Array(buf);
        for (let i = 0; i < raw.length; i++) u8[i] = raw.charCodeAt(i);

        if (ws && ws.readyState === WebSocket.OPEN) {
          sendAudioToSimli(buf);
        } else {
          setStatus("CONNECTING…", "");
          try {
            const token = simliTokenRef.current || (
              await fetch(API.startSession, { method: "POST" }) // ← API
                .then(r => r.json())
                .then(d => d.simli_session_token)
            );
            await connectSimli(token);
            sendAudioToSimli(buf);
          } catch (connErr) {
            console.error("Simli reconnect failed:", connErr);
            setStatus("ERROR — RETRY", "error");
            playPCMLocally(buf);
          }
        }
      }
    } catch (e) {
      addMessage("⚠ Error connecting to the server. Please try again.", "error");
      console.error("Chat error:", e);
    } finally {
      setIsSending(false);
    }
  }, [isSending, inputText, history, connectSimli]);

  // ── Mic toggle ─────────────────────────────────────────────
  const toggleMic = useCallback(async () => {
    if (mediaRecRef.current && mediaRecRef.current.state === "recording") {
      mediaRecRef.current.stop();
      setIsRecording(false);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunks.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecRef.current = recorder;

      recorder.ondataavailable = (e) => audioChunks.current.push(e.data);
      recorder.onstop = async () => {
        stream.getTracks().forEach(t => t.stop());
        const blob     = new Blob(audioChunks.current, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", blob, "voice.webm");
        try {
          const res  = await fetch(API.transcribe, { method: "POST", body: formData }); // ← API
          const data = await res.json();
          if (data.unsupported) {
            addMessage(data.text, "bot");
            return;
          }
          if (data.text) sendMessage(data.text);
        } catch (e) { console.error("Transcription error:", e); }
      };

      recorder.start();
      setIsRecording(true);
    } catch (e) { alert("Microphone access denied."); }
  }, [sendMessage]);

  // ── Lang toggle ────────────────────────────────────────────
  function toggleLang() {
    setLang(prev => prev === "en" ? "ur" : "en");
  }

  const statusColor =
    simliStatus.state === "connected" ? COLORS.green :
    simliStatus.state === "error"     ? COLORS.red   :
    COLORS.gold;

  // ─────────────────────────────────────────────────────────────
  // Render
  // ─────────────────────────────────────────────────────────────
  return (
    <div style={{
      display: "flex", height: "100vh", width: "100vw", overflow: "hidden",
      position: "fixed", top: 0, left: 0,
      fontFamily: "'Segoe UI', sans-serif",
      background: COLORS.cream, color: COLORS.text,
    }}>

      {/* ── Avatar panel ─────────────────────────────────── */}
      <div style={{
        width: "clamp(260px, 32vw, 480px)", minWidth: 260, background: COLORS.greenDark,
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        gap: 20, padding: 32,
      }}>
        <div style={{
          width: 360, height: 360, borderRadius: "50%",
          border: `4px solid ${COLORS.gold}`,
          overflow: "hidden", background: COLORS.greenMid,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            style={{
              width: "100%", height: "100%",
              objectFit: "cover", borderRadius: "50%",
              display: videoVisible ? "block" : "none",
            }}
          />
          {!videoVisible && <AvatarPlaceholder />}
        </div>

        <audio ref={audioRef} autoPlay style={{ display: "none" }} />

        <div style={{ color: "white", fontSize: 18, fontWeight: 700, letterSpacing: 3, textAlign: "center" }}>
          NADRA AI
        </div>
        <div style={{ color: COLORS.gold, fontSize: 11, letterSpacing: 2, textAlign: "center" }}>
          NATIONAL IDENTITY AUTHORITY
        </div>
        <div
          onClick={() => { closeSimli(); stopSilence(); setVideoVisible(false); initSimli(); }}
          style={{ color: statusColor, fontSize: 10, letterSpacing: 1.5, textTransform: "uppercase", cursor: "pointer" }}
        >
          {simliStatus.text}
        </div>
      </div>

      {/* ── Chat panel ───────────────────────────────────── */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>

        <div style={{
          padding: "18px 24px", borderBottom: "1px solid #ddd",
          display: "flex", alignItems: "center", justifyContent: "space-between",
        }}>
          {/* <h1 style={{ fontSize: 18, fontWeight: 600 }}>NADRA Assistant</h1> */}
          {/* <div style={{ fontSize: 11, color: "#888", letterSpacing: 1 }}>SIMLI AI INTERFACE</div> */}
        </div>

        <div style={{
          flex: 1, overflowY: "auto", padding: "20px 24px",
          display: "flex", flexDirection: "column", gap: 12,
        }}>
          {messages.map((msg, i) => (
            <ChatMessage key={i} msg={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        <div style={{
          padding: "16px 24px", borderTop: "1px solid #ddd",
          display: "flex", gap: 10, alignItems: "center", background: "white",
        }}>
          <button
            onClick={toggleMic}
            title="Voice input"
            style={{
              width: 40, height: 40, borderRadius: "50%", border: "none", flexShrink: 0,
              background: isRecording ? COLORS.red : "#f0f0f0",
              cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
            }}
          >
            <MicIcon color={isRecording ? "white" : "#555"} />
          </button>

          <input
            value={inputText}
            onChange={e => setInputText(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter") sendMessage(); }}
            placeholder={lang === "ur" ? "اپنا سوال یہاں لکھیں…" : "Type your question in English or Urdu…"}
            dir={lang === "ur" ? "rtl" : "ltr"}
            style={{
              flex: 1, height: 40, border: "1px solid #ddd", borderRadius: 20,
              padding: "0 16px", fontSize: 14, outline: "none",
              fontFamily: "'Segoe UI', sans-serif",
            }}
          />

          <button
            onClick={toggleLang}
            style={{
              width: 48, height: 40, borderRadius: 20,
              border: "1px solid #ddd", background: "white",
              fontSize: 12, fontWeight: 700, cursor: "pointer", flexShrink: 0,
              display: "flex", alignItems: "center", justifyContent: "center",
              color: COLORS.greenDark,
            }}
          >
            {lang ? lang.toUpperCase() : "EN"}
          </button>

          <button
            onClick={() => sendMessage()}
            disabled={isSending}
            style={{
              width: 40, height: 40, borderRadius: "50%", border: "none", flexShrink: 0,
              background: COLORS.greenDark, color: "white",
              cursor: isSending ? "default" : "pointer",
              opacity: isSending ? 0.5 : 1,
              display: "flex", alignItems: "center", justifyContent: "center",
            }}
          >
            <SendIcon />
          </button>
        </div>
      </div>
    </div>
  );
}