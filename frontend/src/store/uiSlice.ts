import { createSlice, type PayloadAction } from '@reduxjs/toolkit'
import { searchStarted } from './searchSlice'

export type ResultTab = 'results' | 'pipeline'

interface UiState {
  activeTab: ResultTab
  settingsOpen: boolean
  selectedPatentId: string | null
}

const initialState: UiState = {
  activeTab: 'results',
  settingsOpen: false,
  selectedPatentId: null,
}

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    tabChanged(state, action: PayloadAction<ResultTab>) {
      state.activeTab = action.payload
    },
    settingsToggled(state, action: PayloadAction<boolean>) {
      state.settingsOpen = action.payload
    },
    patentSelected(state, action: PayloadAction<string | null>) {
      state.selectedPatentId = action.payload
    },
  },
  extraReducers: (builder) => {
    // A new search closes any open patent from the previous run.
    builder.addCase(searchStarted, (state) => {
      state.selectedPatentId = null
    })
  },
})

export const { tabChanged, settingsToggled, patentSelected } = uiSlice.actions
export default uiSlice.reducer
