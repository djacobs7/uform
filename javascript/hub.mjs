import { join, dirname } from "path"
import { createWriteStream, existsSync, mkdirSync, writeFileSync } from "fs";

import { downloadFile, listFiles } from "@huggingface/hub";



const Modality = {
    TextEncoder: "text_encoder",
    ImageEncoder: "image_encoder",
    VideoEncoder: "video_encoder",
    TextDecoder: "text_decoder",
};

function isModality(value) {
    return Object.values(Modality).includes(value);
}

function normalizeModalities(modalities) {
    return modalities.map(x => {
        if (typeof x === "string") {
            if (isModality(x)) {
                return x;
            } else {
                throw new Error(`Invalid modality: ${x}`);
            }
        }
        return x;
    });
}

async function ensureDirectoryExists(dirPath) {
    if (!existsSync(dirPath)) {
        mkdirSync(dirPath, { recursive: true });
    }
}

async function getModel(modelId, modalities, token = null, format = '.onnx', saveDir = './models') {
    modalities = normalizeModalities(modalities);

    const configNames = ['config.json'];
    const tokenizerNames = ['tokenizer.json'];
    const modelFileNames = modalities.map(modality => `${modality}${format}`);



    const allowedPatterns = [...modelFileNames, ...configNames, ...tokenizerNames];



    const repo = { type: "model", name: modelId };
    const credentials = token ? { accessToken: token } : undefined;

    let configPath = null;
    let tokenizerPath = null;
    const modalityPaths = {};
    const modelSaveDir = join(saveDir, modelId);

    await ensureDirectoryExists(modelSaveDir);

    const fileIterator = listFiles({ repo, recursive: true, credentials });
    for await (const file of fileIterator) {
        const fileName = file.path.split('/').pop();
        if (fileName) {
            const matchingPattern = allowedPatterns.find(pattern => {
                // Convert glob pattern to regexp
                const regexPattern = pattern.replace(/\./g, '\\.').replace(/\*/g, '.*');
                return new RegExp(`${regexPattern}`).test(file.path);
            });

            if (matchingPattern) {

                const filePath = file.path;
                const savePath = join(modelSaveDir, file.path);

                // Create directory for savePath if it doesn't exist
                const saveDir = dirname(savePath);
                await ensureDirectoryExists(saveDir);

                if (configNames.includes(fileName)) {
                    configPath = savePath;
                } else if (tokenizerNames.includes(fileName)) {
                    tokenizerPath = savePath;
                } else {
                    const modalityName = fileName.split('.')[0];
                    modalityPaths[modalityName] = savePath;
                }

                // Skip download if file already exists
                if (existsSync(savePath)) {
                    console.log(`File ${fileName} already exists at ${savePath}, skipping download`);
                    continue;
                }

                const response = await downloadFile({ repo, path: filePath, credentials });
                if (response) {
                    if (response.body && response.body.pipe) {
                        const fileStream = createWriteStream(savePath);
                        response.body.pipe(fileStream);
                        await new Promise((resolve, reject) => {
                            fileStream.on('finish', resolve);
                            fileStream.on('error', reject);
                        });
                    } else if (response.arrayBuffer) {
                        // Handle non-streamable response for environments like Node.js
                        const buffer = await response.arrayBuffer();
                        writeFileSync(savePath, Buffer.from(buffer));
                    } else {
                        console.error('Unexpected response type');
                    }
                    console.log(`Downloaded ${fileName} successfully to ${savePath}`);
                } else {
                    console.log('No response received for the file download request.');
                }
            }
        }
    }

    return { configPath, modalityPaths, tokenizerPath };
}

export { getModel, Modality };
