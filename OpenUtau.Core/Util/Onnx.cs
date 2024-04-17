using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenUtau.Core.Util;
// using Vortice.DXGI;

namespace OpenUtau.Core {
    public class GpuInfo {
        public int deviceId;
        public string description = "";

        override public string ToString() {
            return $"[{deviceId}] {description}";
        }
    }

    public class Onnx {
        public static List<string> getRunnerOptions() {
            if (OS.IsWindows()) {
                return new List<string> {
                "cpu",
                // "directml",
                "cuda"
                };
            } else if (OS.IsMacOS()) {
                return new List<string> {
                "cpu",
                "coreml"
                };
            } else if (OS.IsLinux()) {
                return new List<string> {
                "cpu",
                "cuda"
                };
            }
            return new List<string> {
                "cpu"
            };
        }

        public static List<GpuInfo> getGpuInfo() {
            List<GpuInfo> gpuList = new List<GpuInfo>();
            // if (OS.IsWindows()) {
            //     DXGI.CreateDXGIFactory1(out IDXGIFactory1 factory);
            //     for(int deviceId = 0; deviceId < 32; deviceId++) {
            //         factory.EnumAdapters1(deviceId, out IDXGIAdapter1 adapterOut);
            //         if(adapterOut is null) {
            //             break;
            //         }
            //         gpuList.Add(new GpuInfo {
            //             deviceId = deviceId,
            //             description = adapterOut.Description.Description
            //         }) ;
            //     }
            // }
            if (OS.IsWindows() || OS.IsLinux()) {
                var psi = new ProcessStartInfo {
                    FileName = "nvidia-smi",
                    Arguments = "--query-gpu=index,name --format=csv,noheader",
                    RedirectStandardOutput = true,
                    UseShellExecute = false
                };

                var process = Process.Start(psi);
                if (process != null) {
                    string output = process.StandardOutput.ReadToEnd();
                    string[] lines = output.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

                    foreach (string line in lines) {
                        string[] parts = line.Split(',', 2);
                        if (parts.Length < 2) {
                            continue;
                        }

                        if (!int.TryParse(parts[0].Trim(), out int deviceId)) {
                            continue;
                        }

                        string deviceName = parts[1].Trim();
                        gpuList.Add(new GpuInfo {
                            deviceId = deviceId,
                            description = deviceName
                        });
                    }
                }
            }
            if (gpuList.Count == 0) {
                gpuList.Add(new GpuInfo {
                    deviceId = 0,
                });
            }
            return gpuList;
        }

        private static SessionOptions getOnnxSessionOptions(){
            SessionOptions options = new SessionOptions();
            List<string> runnerOptions = getRunnerOptions();
            string runner = Preferences.Default.OnnxRunner;
            if (String.IsNullOrEmpty(runner)) {
                runner = runnerOptions[0];
            }
            if (!runnerOptions.Contains(runner)) {
                runner = "cpu";
            }
            switch(runner){
                case "directml":
                    options.AppendExecutionProvider_DML(Preferences.Default.OnnxGpu);
                    break;
                case "coreml":
                    options.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ENABLE_ON_SUBGRAPH);
                    break;
                case "cuda": {
                    var cudaOptions = new OrtCUDAProviderOptions();
                    cudaOptions.UpdateOptions(new Dictionary<string, string> {
                        ["device_id"] = Preferences.Default.OnnxGpu.ToString(),
                        ["cudnn_conv_algo_search"] = "DEFAULT"
                    });
                    options.AppendExecutionProvider_CUDA(cudaOptions);
                    break;
                }
            }
            return options;
        }

        public static InferenceSession getInferenceSession(byte[] model) {
            return new InferenceSession(model,getOnnxSessionOptions());
        }

        public static InferenceSession getInferenceSession(string modelPath) {
            return new InferenceSession(modelPath,getOnnxSessionOptions());
        }

        public static void VerifyInputNames(InferenceSession session, IEnumerable<NamedOnnxValue> inputs) {
            var sessionInputNames = session.InputNames.ToHashSet();
            var givenInputNames = inputs.Select(v => v.Name).ToHashSet();
            var missing = sessionInputNames
                .Except(givenInputNames)
                .OrderBy(s => s, StringComparer.InvariantCulture)
                .ToArray();
            if (missing.Length > 0) {
                throw new ArgumentException("Missing input(s) for the inference session: " + string.Join(", ", missing));
            }
            var unexpected = givenInputNames
                .Except(sessionInputNames)
                .OrderBy(s => s, StringComparer.InvariantCulture)
                .ToArray();
            if (unexpected.Length > 0) {
                throw new ArgumentException("Unexpected input(s) for the inference session: " + string.Join(", ", unexpected));
            }
        }
    }
}
