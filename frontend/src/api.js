// src/api.js
const BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const API = {
  startSession: `${BASE_URL}/start_session`,
  chat:         `${BASE_URL}/chat`,
  transcribe:   `${BASE_URL}/transcribe`,
};