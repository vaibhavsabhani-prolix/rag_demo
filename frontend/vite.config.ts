import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, 'src') },
  },
  server: {
    // Forward API calls to the FastAPI server (see app/api/main.py).
    // In Docker the dev server reaches it by service name (docker-compose.override.yml).
    proxy: { '/api': process.env.API_PROXY_TARGET ?? 'http://localhost:8000' },
    // Let ngrok or Cloudflare tunnels reach the dev server.
    allowedHosts: ['.ngrok-free.app', '.ngrok-free.dev', '.ngrok.app', '.ngrok.dev', '.trycloudflare.com', '.loca.lt'],
  },
  preview: {
    host: '0.0.0.0',
    port: 5173,
    proxy: { '/api': process.env.API_PROXY_TARGET ?? 'http://localhost:8000' },
    allowedHosts: ['.ngrok-free.app', '.ngrok-free.dev', '.ngrok.app', '.ngrok.dev', '.trycloudflare.com', '.loca.lt'],
  },
  build: {
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
  },
})
