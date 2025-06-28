const ContextSelect = document.getElementById("context_type");
const TextContextInput = document.getElementById("context-text-input");
const FileContextInput = document.getElementById("context-file-input");

function handleContextSelect() {
  const Value = ContextSelect.value;

  // Reset all inputs to hidden
  TextContextInput.style.display = "none";
  FileContextInput.style.display = "none";

  // Show the selected input
  if (Value === "text") {
    TextContextInput.style.display = "block";
  } else if (Value === "file") {
    FileContextInput.style.display = "block";
  }
}

// Set initial state on page load
document.addEventListener("DOMContentLoaded", handleContextSelect);

// Handle dropdown change
ContextSelect.addEventListener("change", handleContextSelect);
