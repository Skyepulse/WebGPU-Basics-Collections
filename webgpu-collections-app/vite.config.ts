import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'
import { mkdir, copyFile } from 'fs/promises'

const LEVEL_COUNT = 14

function emitStaticLevelEntryPoints() {
  let rootDir = process.cwd()
  let outDir = 'dist'

  return {
    name: 'emit-static-level-entry-points',
    configResolved(config: { root: string; build: { outDir: string } }) {
      rootDir = config.root
      outDir = config.build.outDir
    },
    async closeBundle() {
      const resolvedOutDir = path.resolve(rootDir, outDir)
      const sourceIndex = path.join(resolvedOutDir, 'index.html')

      for (let level = 1; level <= LEVEL_COUNT; level++) {
        const levelDir = path.join(resolvedOutDir, String(level))
        await mkdir(levelDir, { recursive: true })
        await copyFile(sourceIndex, path.join(levelDir, 'index.html'))
      }
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  base: '/',
  plugins: [vue(), tailwindcss(), emitStaticLevelEntryPoints()],
  resolve : {
    alias: {
      '@src': path.resolve(__dirname, './src'),
      '@assets': path.resolve(__dirname, './src/assets'),
    },
  },
})
