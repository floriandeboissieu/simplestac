// Get a NodeList of all .demo elements
const demoClasses = document.querySelectorAll('.go');

// Change the text of multiple elements with a loop
demoClasses.forEach(element => {
  element.classList.replace('go', 'language-text');
});