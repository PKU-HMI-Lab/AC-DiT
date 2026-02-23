window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function () {
	// Check for click events on the navbar burger icon

	var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
	}

	// Initialize all div with carousel class
	var carousels = bulmaCarousel.attach('.carousel', options);

	bulmaSlider.attach();
	document.getElementById('copyButton').addEventListener('click', copyText);

	// Add scroll handler for tables
	const tables = document.querySelectorAll('table');
	tables.forEach(table => {
		updateTableShadows(table);
		table.addEventListener('scroll', () => updateTableShadows(table));
	});
})

function updateTableShadows(table) {
	const maxScroll = table.scrollWidth - table.clientWidth;
	const scrollLeft = table.scrollLeft;
	
	let background = [];
	
	// Add left shadow if not at start
	if (scrollLeft > 0) {
		background.push('radial-gradient(50% 50% at 0 50%, rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0))');
	}
	
	// Add right shadow if not at end
	if (scrollLeft < maxScroll) {
		background.push('radial-gradient(50% 50% at 100% 50%, rgba(0, 0, 0, 0.35), rgba(0, 0, 0, 0)) 100% 0');
	}
	
	// Always maintain the white gradients for smooth transitions
	background.unshift(
		'linear-gradient(to right, white 30%, rgba(255, 255, 255, 0))',
		'linear-gradient(to right, rgba(255, 255, 255, 0), white 70%) 100% 0'
	);
	
	table.style.background = background.join(',');
	table.style.backgroundRepeat = 'no-repeat';
	table.style.backgroundSize = '60px 100%';
	table.style.backgroundAttachment = 'local, local, scroll, scroll';
}

const copyText = () => {
	const text = `@article{shukla2024maniskillhab,
	author		 = {Arth Shukla and Stone Tao and Hao Su},
	title        = {ManiSkill-HAB: A Benchmark for Low-Level Manipulation in Home Rearrangement Tasks},
	journal      = {CoRR},
	volume       = {abs/2412.13211},
	year         = {2024},
	url          = {https://doi.org/10.48550/arXiv.2412.13211},
	doi          = {10.48550/ARXIV.2412.13211},
	eprinttype   = {arXiv},
	eprint       = {2412.13211},
	timestamp    = {Mon, 09 Dec 2024 01:29:24 +0100},
	biburl       = {https://dblp.org/rec/journals/corr/abs-2412-13211.bib},
	bibsource    = {dblp computer science bibliography, https://dblp.org}
}`;

	navigator.clipboard.writeText(text).then(() => {
		const messageDiv = document.getElementById('message');
		messageDiv.textContent = 'Copied to clipboard';
		messageDiv.style.color = '#5a5aFa';
		messageDiv.style.paddingRight = '10px';

		setTimeout(() => {
			messageDiv.textContent = '';
		}, 2000);
	}).catch(err => {
		console.error('Failed to copy: ', err);
	});
};

