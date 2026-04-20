<template>
  <div class="home-container">
    <!-- Top Navigation Bar -->
    <nav class="navbar" :style="s.navbar">
      <div class="nav-brand" :style="s.navBrand">MIROFISH <span class="nav-brand-version">OFFLINE</span></div>
      <div class="nav-links" :style="s.navLinks">
        <a href="https://github.com/nikmcfly/MiroFish-Offline" target="_blank" class="github-link" :style="s.githubLink">
          Visit our Github <span>↗</span>
        </a>
      </div>
    </nav>

    <div class="main-content" :style="s.mainContent">
      <!-- Viewport-height centering wrapper -->
      <div class="viewport-center">
      <!-- Dashboard: Two-Column Layout -->
      <section class="dashboard-section" :style="s.dashboardSection">
        <!-- Left Column: Status & Steps -->
        <div class="left-panel" :style="s.leftPanel">
          <div class="panel-header" :style="s.panelHeader">
            <span :style="s.statusDot">■</span> System Status
          </div>

          <h2 class="section-title" :style="s.sectionTitle">Ready</h2>
          <p class="section-desc" :style="s.sectionDesc">
            Local prediction engine on standby. Upload unstructured data to initialize a simulation.
          </p>

          <div class="steps-container" :style="s.stepsContainer">
            <div class="steps-header" :style="s.stepsHeader">
               <span :style="s.diamondIcon">◇</span> Workflow Sequence
            </div>
            <div class="workflow-list" :style="s.workflowList">
              <div v-for="(step, i) in steps" :key="i" class="workflow-item" :style="s.workflowItem">
                <span class="step-num" :style="s.stepNum">{{ step.num }}</span>
                <div class="step-info" :style="s.stepInfo">
                  <div class="step-title" :style="s.stepTitle">{{ step.title }}</div>
                  <div class="step-desc" :style="s.stepDesc">{{ step.desc }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Right Column: Interactive Console -->
        <div class="right-panel" :style="s.rightPanel">
          <div class="console-box" :style="s.consoleBox">
            <div :style="s.consoleSection">
              <div class="console-header" :style="s.consoleHeader">
                <span>01 / Reality Seeds</span>
                <span>PDF · MD · TXT</span>
              </div>
              <div
                :style="s.uploadZone"
                :class="{ 'is-drag-over': isDragOver }"
                @dragover.prevent="handleDragOver"
                @dragleave.prevent="handleDragLeave"
                @drop.prevent="handleDrop"
                @click="triggerFileInput"
              >
                <input ref="fileInput" type="file" multiple accept=".pdf,.md,.txt" @change="handleFileSelect" style="display: none" :disabled="loading" />
                <div v-if="files.length === 0" :style="s.uploadPlaceholder">
                  <div :style="s.uploadIcon">↑</div>
                  <div :style="s.uploadTitle">Drop files to begin</div>
                  <div :style="s.uploadHint">PDF · MD · TXT</div>
                </div>
                <div v-else :style="s.fileList">
                  <div v-for="(file, index) in files" :key="index" :style="s.fileItem">
                    <span>📄</span>
                    <span :style="s.fileName">{{ file.name }}</span>
                    <button @click.stop="removeFile(index)" :style="s.removeBtn">×</button>
                  </div>
                </div>
              </div>
            </div>

            <div :style="s.consoleDivider"><span :style="s.consoleDividerText">Parameters</span></div>

            <div :style="s.consoleSection">
              <div class="console-header" :style="s.consoleHeader">
                <span>>_ 02 / Simulation Prompt</span>
              </div>
              <div :style="s.inputWrapper">
                <textarea v-model="formData.simulationRequirement" :style="s.codeInput" placeholder="// Describe your simulation or prediction goal in natural language" rows="6" :disabled="loading"></textarea>
                <div :style="s.modelBadge">Engine: Ollama + Neo4j (local)</div>
              </div>
            </div>

            <div :style="s.btnSection">
              <button :style="s.startEngineBtn" @click="startSimulation" :disabled="!canSubmit || loading">
                <span v-if="!loading">Start Engine</span>
                <span v-else>Initializing...</span>
                <span>→</span>
              </button>
            </div>
          </div>
        </div>
      </section>
      </div>

      <HistoryDatabase />
    </div>
  </div>
</template>

<script setup>
import './Home.css'
import { ref, computed, reactive } from 'vue'
import { useRouter } from 'vue-router'
import HistoryDatabase from '../components/HistoryDatabase.vue'

const mono = 'JetBrains Mono, monospace'
const sans = 'Space Grotesk, Noto Sans SC, system-ui, sans-serif'

const s = reactive({
  navbar: { height: '56px', background: '#111110', color: '#F5F0EB', display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 40px' },
  navBrand: { fontFamily: mono, fontWeight: '600', letterSpacing: '0.08em', fontSize: '0.85rem', color: '#F5F0EB', display: 'flex', alignItems: 'center', gap: '10px' },
  navLinks: { display: 'flex', alignItems: 'center' },
  githubLink: { color: '#8C8580', textDecoration: 'none', fontFamily: mono, fontSize: '0.75rem', fontWeight: '400', display: 'flex', alignItems: 'center', gap: '6px', padding: '6px 12px', border: '1px solid transparent', letterSpacing: '0.04em' },
  mainContent: { maxWidth: '1400px', margin: '0 auto' },
  dashboardSection: { display: 'flex', gap: '52px', paddingTop: '0', alignItems: 'flex-start', borderTop: 'none' },
  leftPanel: { flex: '0.8', display: 'flex', flexDirection: 'column' },
  panelHeader: { fontFamily: mono, fontSize: '0.7rem', color: '#9C948A', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '12px', letterSpacing: '0.05em' },
  statusDot: { color: '#4A7A62', fontSize: '0.75rem' },
  sectionTitle: { fontSize: '1.8rem', fontWeight: '500', margin: '0 0 10px 0', letterSpacing: '-0.02em', color: '#1A1816' },
  sectionDesc: { color: '#6B6560', marginBottom: '28px', lineHeight: '1.65', fontSize: '0.9rem' },
  stepsContainer: { background: '#F4F2EF', padding: '24px', position: 'relative' },
  stepsHeader: { fontFamily: mono, fontSize: '0.7rem', color: '#9C948A', marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '8px', letterSpacing: '0.06em', textTransform: 'uppercase' },
  diamondIcon: { fontSize: '1rem', lineHeight: '1' },
  workflowList: { display: 'flex', flexDirection: 'column', gap: '16px' },
  workflowItem: { display: 'flex', alignItems: 'flex-start', gap: '16px', position: 'relative' },
  stepNum: { fontFamily: mono, fontWeight: '600', color: '#1A1816', fontSize: '0.7rem', paddingTop: '2px', minWidth: '20px' },
  stepInfo: { flex: '1' },
  stepTitle: { fontWeight: '600', fontSize: '0.875rem', marginBottom: '2px', color: '#1A1816' },
  stepDesc: { fontSize: '0.8rem', color: '#9C948A', lineHeight: '1.5' },
  rightPanel: { flex: '1.2', display: 'flex', flexDirection: 'column' },
  consoleBox: { border: '1px solid #C8C0B8', background: '#FFFFFF' },
  consoleSection: { padding: '20px 20px 16px' },
  consoleHeader: { display: 'flex', justifyContent: 'space-between', marginBottom: '12px', fontFamily: mono, fontSize: '0.7rem', color: '#9C948A', letterSpacing: '0.04em' },
  uploadZone: { border: '1.5px dashed #C8C0B8', height: '160px', overflowY: 'auto', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', background: '#F4F2EF', transition: 'border-color 150ms ease, background 150ms ease' },
  uploadPlaceholder: { textAlign: 'center' },
  uploadIcon: { width: '34px', height: '34px', border: '1px solid #DDD8D0', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 12px', color: '#9C948A', background: '#fff' },
  uploadTitle: { fontWeight: '500', fontSize: '0.875rem', marginBottom: '4px', color: '#1A1816' },
  uploadHint: { fontFamily: mono, fontSize: '0.7rem', color: '#B0A89E', letterSpacing: '0.04em' },
  fileList: { width: '100%', padding: '12px', display: 'flex', flexDirection: 'column', gap: '8px' },
  fileItem: { display: 'flex', alignItems: 'center', background: '#fff', padding: '7px 10px', border: '1px solid #EAE6E0', fontFamily: mono, fontSize: '0.8rem' },
  fileName: { flex: '1', margin: '0 10px', color: '#1A1816' },
  removeBtn: { background: 'none', border: 'none', cursor: 'pointer', fontSize: '1.1rem', color: '#9C948A' },
  consoleDivider: { display: 'flex', alignItems: 'center', borderTop: '1px solid #EAE6E0', padding: '8px 20px', margin: '0' },
  consoleDividerText: { fontFamily: mono, fontSize: '0.65rem', color: '#B0A89E', letterSpacing: '0.08em', textTransform: 'uppercase' },
  inputWrapper: { position: 'relative', border: '1px solid #DDD8D0', background: '#F4F2EF' },
  codeInput: { width: '100%', border: 'none', background: 'transparent', padding: '16px', fontFamily: mono, fontSize: '0.875rem', lineHeight: '1.7', resize: 'none', outline: 'none', minHeight: '140px', color: '#1A1816' },
  modelBadge: { position: 'absolute', bottom: '10px', right: '12px', fontFamily: mono, fontSize: '0.65rem', color: '#B0A89E', background: '#EDE9E4', padding: '2px 6px', letterSpacing: '0.03em' },
  btnSection: { padding: '0 16px 16px' },
  startEngineBtn: { width: '100%', background: '#1A1816', color: '#F5F0EB', border: '1px solid #2A2826', padding: '18px 24px', fontFamily: mono, fontWeight: '600', fontSize: '0.875rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', letterSpacing: '0.08em', transition: 'background 150ms ease, border-color 150ms ease' },
})

const steps = [
  { num: '01', title: 'Graph Build', desc: 'Extract reality seeds from your document, build knowledge graph with Neo4j + GraphRAG' },
  { num: '02', title: 'Env Setup', desc: 'Generate agent personas, configure simulation parameters via local Ollama LLM' },
  { num: '03', title: 'Simulation', desc: 'Run multi-agent simulation locally with dynamic memory updates and emergent behavior' },
  { num: '04', title: 'Report', desc: 'ReportAgent analyzes the simulation results and generates a detailed prediction report' },
  { num: '05', title: 'Interaction', desc: 'Chat with any agent from the simulated world or discuss findings with ReportAgent' },
]

const router = useRouter()

const formData = ref({ simulationRequirement: '' })
const files = ref([])
const loading = ref(false)
const error = ref('')
const isDragOver = ref(false)
const fileInput = ref(null)

const canSubmit = computed(() => {
  return formData.value.simulationRequirement.trim() !== '' && files.value.length > 0
})

const triggerFileInput = () => { if (!loading.value) fileInput.value?.click() }
const handleFileSelect = (event) => { addFiles(Array.from(event.target.files)) }
const handleDragOver = (e) => { isDragOver.value = true }
const handleDragLeave = (e) => { isDragOver.value = false }
const handleDrop = (e) => { isDragOver.value = false; addFiles(Array.from(e.dataTransfer.files)) }

const addFiles = (newFiles) => {
  const allowed = ['.pdf', '.md', '.txt']
  const valid = newFiles.filter(f => allowed.some(ext => f.name.toLowerCase().endsWith(ext)))
  files.value = [...files.value, ...valid]
}

const removeFile = (index) => { files.value.splice(index, 1) }

const startSimulation = () => {
  if (!canSubmit.value || loading.value) return
  import('../store/pendingUpload.js').then(({ setPendingUpload }) => {
    setPendingUpload(files.value, formData.value.simulationRequirement)
    router.push({ name: 'Process', params: { projectId: 'new' } })
  })
}
</script>

<!-- Styles loaded from Home.css via import -->
