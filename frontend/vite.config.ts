import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Fixed port (must match CORS_ORIGINS on the backend); fail instead of silently picking another.
  server: { port: 5180, strictPort: true },
  preview: { port: 5180, strictPort: true },
})
