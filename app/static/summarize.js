const SelectValue = document.getElementById("input_type");
const Text_Input = document.getElementById("Text-input");
const Url_Input = document.getElementById("Url-input");
const File_Input = document.getElementById("file-input");

function handleSelect() {
  const Value = SelectValue.value;

  // Reset all inputs to hidden
  Text_Input.style.display = "none";
  Url_Input.style.display = "none";
  File_Input.style.display = "none";

  // Show the selected input
  if (Value === "text") {
    Text_Input.style.display = "block";
  } else if (Value === "url") {
    Url_Input.style.display = "block";
  } else if (Value === "pdf") {
    File_Input.style.display = "block";
  }
}

// Set initial state on page load
document.addEventListener("DOMContentLoaded", handleSelect);

// Handle dropdown change
SelectValue.addEventListener("change", handleSelect);
