import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // Load environment variables from the root directory
  envDir: path.resolve(__dirname, '../'),
  // Expose SUPABASE_ variables to the client
  envPrefix: ['VITE_', 'SUPABASE_'],
  server: {
    port: 3000,
    open: true
  }
})
