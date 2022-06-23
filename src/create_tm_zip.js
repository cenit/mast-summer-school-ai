
import { AudioContext } from 'web-audio-api'

const tools= require("./tools.js")

let c = new AudioContext({
    sampleRate: 44100,
});
//let b = fetch(`${$input.value}.wav`)
let b = fetch(`../res/Ensoniq-ZR-76-01-Dope-77.wav`)
    .then((response) => response.arrayBuffer())
    .then((buffer) => c.decodeAudioData(buffer));

let freqDataQueue = [];
let columnTruncateLength = 232;
let sampleRate = 44100;

let oac = new OfflineAudioContext({
    numberOfChannels: b.numberOfChannels,
    length: b.length,
    sampleRate: sampleRate,
});

const source = oac.createBufferSource();
const processor = oac.createScriptProcessor(1024, 1, 1);

const analyser = oac.createAnalyser();
analyser.fftSize = 2048;
analyser.smoothingTimeConstant = 0;

source.buffer = b;

source.connect(analyser);
analyser.connect(processor);
processor.connect(oac.destination);

var freqData = new Float32Array(analyser.fftSize);
processor.onaudioprocess = () => {
    analyser.getFloatFrequencyData(freqData);
    freqDataQueue.push(freqData.slice(0, columnTruncateLength));
};

source.start(0);
oac.startRendering();

oac.oncomplete = (e) => {
    console.log(freqDataQueue);
    source.disconnect(analyser);
    processor.disconnect(oac.destination);
};
