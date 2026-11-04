/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: { port: 5173, strictPort: true },
  build: { target: "es2022", outDir: "dist" },
  // The app's tests only: scripts/release/test holds the release scripts' tests, which run under node:test (npm run test:release).
  test: { environment: "jsdom", globals: true, reporters: ["default"], include: ["src/**/*.test.{ts,tsx}"] },
});
