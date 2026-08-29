/**
 * ResidualPyramid.mjs
 * 残差金字塔模块 (Physics-Optimized Edition)
 * 功能：基于 Gram-Schmidt 正交化计算多层级语义残差，精确分析语义能量谱。
 *
 * ── 移植声明 (VCPToolBox, CC BY-NC-SA 4.0) ──────────────────────────
 * 本文件移植自 VCPToolBox (https://github.com/lioensky/VCPToolBox) 的
 * `ResidualPyramid.js` (CommonJS, 394 行), 遵循 CC BY-NC-SA 4.0 许可,
 * 版权归原作者所有。移植日期: 2026-08-29。
 * 移植规则: 数学代码(Modified Gram-Schmidt 正交投影 / 握手差值分析 /
 *   能量截断 / 特征提取)逐行保留, 仅做外部接口替换:
 *   1) CommonJS (module.exports) → ESM (export class);
 *   2) 构造参数 (tagIndex, db) → searchTags 回调:
 *        async (residualVector: Float32Array, topK: number)
 *        => Array<{ id:number, name:string, vector:Float32Array, similarity:number }>
 *      (融合原 tagIndex.search + db 取向量两步; 实现方是 taglayer 的暴力余弦);
 *   3) analyze() 变为 async (searchTags 是异步回调);
 *   4) 删除 Rust 快路径分支 (computeOrthogonalProjection / computeHandshakes), 仅保留 JS 实现;
 *   5) 删除 tag id 的 BigInt 转换逻辑 (本项目的 id 就是 number);
 *   6) 删除原 _getTagVectors (SQLite 按 id 取向量) — 向量已由 searchTags 直接返回。
 * ─────────────────────────────────────────────────────────────────────
 */

export class ResidualPyramid {
    constructor(searchTags, config = {}) {
        // 改动: 原构造参数 (tagIndex, db) 合并为一个搜索回调 searchTags:
        //   async (residualVector: Float32Array, topK: number)
        //   => Array<{ id:number, name:string, vector:Float32Array, similarity:number }>
        this.searchTags = searchTags;
        this.config = {
            maxLevels: config.maxLevels || 3,
            topK: config.topK || 10,
            // 修正：使用能量阈值。0.1 表示当残差能量低于原始能量的 10% 时停止 (即解释了 90%)
            minEnergyRatio: config.minEnergyRatio || 0.1,
            dimension: config.dimension || 3072,
            ...config
        };
    }

    /**
     * 🌟 核心：计算查询向量的残差金字塔
     * @param {Float32Array|Array} queryVector - 原始查询向量
     * @returns {Promise<Object>} 金字塔结果 (改动: 因 searchTags 为异步回调, analyze 变为 async)
     */
    async analyze(queryVector) {
        const dim = this.config.dimension;
        const pyramid = {
            levels: [],
            totalExplainedEnergy: 0, // 被Tag解释的总能量比例 (0~1)
            finalResidual: null,     // 最终残差向量
            features: {}             // 提取的特征
        };

        // 确保使用 Float32Array
        let currentVector = queryVector instanceof Float32Array ? queryVector : new Float32Array(queryVector);

        // 计算初始总能量 E = ||v||^2
        const originalMagnitude = this._magnitude(currentVector);
        const originalEnergy = originalMagnitude * originalMagnitude;

        // 防止除零错误
        if (originalEnergy < 1e-12) {
            return this._emptyResult(dim);
        }

        let currentResidual = new Float32Array(currentVector); // 迭代中的残差

        for (let level = 0; level < this.config.maxLevels; level++) {
            // 1. 搜索当前残差向量的最近 Tags
            let tagResults;
            try {
                // 🚀 改动: searchTags 融合原 tagIndex.search + db 取向量两步,
                // 直接返回 { id, name, vector, similarity }
                tagResults = await this.searchTags(currentResidual, this.config.topK);
            } catch (e) {
                console.warn(`[Residual] Search failed at level ${level}:`, e.message);
                break;
            }

            if (!tagResults || tagResults.length === 0) break;

            // 改动: 删除原 "Rust 返回的 id 是 BigInt, 需转换为 Number" 的逻辑
            // 与本项目自写的 _getTagVectors (SQLite 按 id 取向量) —
            // searchTags 已同时返回向量与相似度, 无需二次取数。
            const rawTags = tagResults;
            if (rawTags.length === 0) break;

            // 3. 🌟 核心修正：Gram-Schmidt 正交投影
            // 计算当前残差在这些 Tag 张成的子空间上的精确投影
            const { projection, residual, orthogonalBasis, basisCoefficients } = this._computeOrthogonalProjection(
                currentResidual, rawTags
            );

            // 4. 计算能量数据
            const residualMagnitude = this._magnitude(residual);
            const residualEnergy = residualMagnitude * residualMagnitude;
            const currentEnergy = this._magnitude(currentResidual) ** 2;

            // 本层解释的能量 = (旧残差能量 - 新残差能量) / 原始总能量
            // 注意：由于正交投影性质，||R_old||^2 = ||Projection||^2 + ||R_new||^2
            const energyExplainedByLevel = Math.max(0, currentEnergy - residualEnergy) / originalEnergy;

            // 5. 分析握手特征 (基于原始 Tag 方向，而非正交基)
            const handshakes = this._computeHandshakes(currentResidual, rawTags);

            pyramid.levels.push({
                level,
                tags: rawTags.map((t, i) => {
                    // 改动: 原代码 `Number(r.id) === t.id` (BigInt→Number),
                    // 项目 id 就是 number, 直接相等比较
                    const res = tagResults.find(r => r.id === t.id);
                    // 估算该 Tag 在本层解释中的贡献度 (基于其在正交基中的投影分量)
                    // 这是一个近似值，因为 Gram-Schmidt 对顺序敏感，但这比单纯的 softmax 准确
                    return {
                        id: t.id,
                        name: t.name,
                        // 改动: 原字段名 score (Rust SearchResult) → similarity (新 searchTags 契约)
                        similarity: res ? res.similarity : 0,
                        // 修正：权重不再是 softmax，而是该 Tag 对解释能量的贡献
                        contribution: basisCoefficients[i] || 0,
                        handshakeMagnitude: handshakes.magnitudes[i]
                    };
                }),
                projectionMagnitude: this._magnitude(projection),
                residualMagnitude,
                residualEnergyRatio: residualEnergy / originalEnergy,
                energyExplained: energyExplainedByLevel,
                handshakeFeatures: this._analyzeHandshakes(handshakes, dim)
            });

            pyramid.totalExplainedEnergy += energyExplainedByLevel;
            currentResidual = residual; // 更新残差用于下一轮

            // 6. 能量阈值截断 (Energy Cutoff)
            // 如果剩余能量少于设定的比例 (例如 10%)，则停止
            if ((residualEnergy / originalEnergy) < this.config.minEnergyRatio) {
                break;
            }
        }

        pyramid.finalResidual = currentResidual;
        pyramid.features = this._extractPyramidFeatures(pyramid);

        return pyramid;
    }

    /**
     * 🌟 数学修正：Gram-Schmidt 正交化投影
     * 将 vector 投影到 tags 张成的子空间中
     */
    _computeOrthogonalProjection(vector, tags) {
        const dim = this.config.dimension;
        const n = tags.length;

        // 改动: Rust 快路径分支 (tagIndex.computeOrthogonalProjection) 已按规格删除,
        // 仅保留下面的 JS 实现 (Modified Gram-Schmidt)。

        const basis = []; // 存储正交基向量 { vec: Float32Array, originalIndex: number }
        const basisCoefficients = new Float32Array(n); // 记录每个 Tag (对应基) 承载的投影分量

        // 1. 构建正交基 (Modified Gram-Schmidt 算法，数值更稳定)
        for (let i = 0; i < n; i++) {
            const tagVec = this._extractFloat32(tags[i].vector);

            // v_i = t_i
            let v = new Float32Array(tagVec);

            // 减去在已有基上的投影: v = v - <v, u_j> * u_j
            for (let j = 0; j < basis.length; j++) {
                const u = basis[j];
                const dot = this._dotProduct(v, u);
                for (let d = 0; d < dim; d++) {
                    v[d] -= dot * u[d];
                }
            }

            // 归一化得到 u_i
            const mag = this._magnitude(v);
            if (mag > 1e-6) { // 防止零向量
                for (let d = 0; d < dim; d++) v[d] /= mag;
                basis.push(v);

                // 计算 Query 在这个新基向量上的投影分量系数
                // coeff = <Query, u_i>
                const coeff = this._dotProduct(vector, v);
                basisCoefficients[i] = Math.abs(coeff); // 记录绝对贡献
            } else {
                basisCoefficients[i] = 0; // 该 Tag 线性相关，无独立贡献
            }
        }

        // 2. 计算总投影 P = Σ <vector, u_i> * u_i
        const projection = new Float32Array(dim);
        for (let i = 0; i < basis.length; i++) {
            const u = basis[i];
            const dot = this._dotProduct(vector, u);
            for (let d = 0; d < dim; d++) {
                projection[d] += dot * u[d];
            }
        }

        // 3. 计算残差 R = vector - P
        const residual = new Float32Array(dim);
        for (let d = 0; d < dim; d++) {
            residual[d] = vector[d] - projection[d];
        }

        return { projection, residual, orthogonalBasis: basis, basisCoefficients };
    }

    /**
     * 计算握手差值（查询与每个Tag的差向量）
     * 保留此逻辑用于分析方向性差异
     */
    _computeHandshakes(query, tags) {
        const dim = this.config.dimension;
        const n = tags.length;

        // 改动: Rust 快路径分支 (tagIndex.computeHandshakes) 已按规格删除,
        // 仅保留下面的 JS 实现。

        const magnitudes = [];
        const directions = [];

        for (let i = 0; i < n; i++) {
            const tagVec = this._extractFloat32(tags[i].vector);
            const delta = new Float32Array(dim);
            let magSq = 0;
            for (let d = 0; d < dim; d++) {
                delta[d] = query[d] - tagVec[d];
                magSq += delta[d] * delta[d];
            }
            const mag = Math.sqrt(magSq);
            magnitudes.push(mag);

            const dir = new Float32Array(dim);
            if (mag > 1e-9) {
                for (let d = 0; d < dim; d++) dir[d] = delta[d] / mag;
            }
            directions.push(dir);
        }
        return { magnitudes, directions };
    }

    /**
     * 分析握手差值的统计特征
     * 优化：更清晰的物理意义
     */
    _analyzeHandshakes(handshakes, dim) {
        const n = handshakes.magnitudes.length;
        if (n === 0) return null;

        // 1. 差值方向的一致性 (Coherence)
        // 如果所有 Tag 都在同一个方向上偏离 Query，说明 Query 有明确的“偏移意图”
        const avgDirection = new Float32Array(dim);
        for (let i = 0; i < n; i++) {
            for (let d = 0; d < dim; d++) avgDirection[d] += handshakes.directions[i][d];
        }
        for (let d = 0; d < dim; d++) avgDirection[d] /= n;

        const directionCoherence = this._magnitude(avgDirection);

        // 2. 内部张力 (Internal Tension / Pattern Strength)
        // Tag 之间的差值方向是否相似？
        let pairwiseSimSum = 0;
        let pairCount = 0;
        // 采样前 5 个两两比较，避免 O(N^2)
        const limit = Math.min(n, 5);
        for (let i = 0; i < limit; i++) {
            for (let j = i + 1; j < limit; j++) {
                pairwiseSimSum += Math.abs(this._dotProduct(handshakes.directions[i], handshakes.directions[j]));
                pairCount++;
            }
        }
        const avgPairwiseSim = pairCount > 0 ? pairwiseSimSum / pairCount : 0;

        return {
            // Coherence 高：Query 在所有 Tag 的"外部" (新领域)
            // Coherence 低：Query 被 Tag 包围在"中间" (已知领域的细节)
            directionCoherence,
            patternStrength: avgPairwiseSim,

            // 🌟 修正公式：
            // 新颖信号：方向一致性高(偏移明确) + 残差大(未被解释) -> 这里只计算方向分量
            noveltySignal: directionCoherence,

            // 噪音信号：方向杂乱无章 (Coherence低) 且 Tag 之间也很乱 (Sim低)
            noiseSignal: (1 - directionCoherence) * (1 - avgPairwiseSim)
        };
    }

    /**
     * 提取综合特征
     */
    _extractPyramidFeatures(pyramid) {
        if (pyramid.levels.length === 0) {
            return { depth: 0, coverage: 0, novelty: 1, coherence: 0, tagMemoActivation: 0 };
        }

        const level0 = pyramid.levels[0];
        const handshake = level0.handshakeFeatures;

        // 覆盖率 = 解释的总能量 (0~1)
        const coverage = Math.min(1.0, pyramid.totalExplainedEnergy);

        // 相干度：第一层召回的 Tags 是否属于同一簇
        const coherence = handshake ? handshake.patternStrength : 0;

        // 🌟 修正：Novelty (新颖度)
        // 真正的"新"，是现有的 Tag 解释不了的部分 (Residual Energy)
        // 加上方向一致性 (说明不仅解释不了，而且偏向一个特定未知方向)
        const residualRatio = 1 - coverage;
        const directionalNovelty = handshake ? handshake.noveltySignal : 0;
        const novelty = (residualRatio * 0.7) + (directionalNovelty * 0.3);

        return {
            depth: pyramid.levels.length,
            coverage,
            novelty,
            coherence,

            // 🌟 综合决策指标：是否激活 TagMemo 增强？
            // 逻辑：如果覆盖率已经很高 (Query很常见)，或者完全是噪音，就不需要太强的 Memo
            // 如果相干性高 (Tag 属于同一类)，且有一定覆盖率，说明找到了正确的"邻域"，此时适合激活
            tagMemoActivation: coverage * coherence * (1 - (handshake?.noiseSignal || 0)),

            // 扩展信号：是否需要去搜索新的 Tag？(当新颖度高时)
            expansionSignal: novelty
        };
    }

    /**
     * 改动: 原 _getTagVectors (SQLite 按 id 取向量) 已删除 —
     * searchTags 回调已直接返回 { id, name, vector, similarity }。

     * 安全提取 Float32Array：兼容 SQLite Buffer 和直接传入的 Float32Array
     */
    _extractFloat32(vectorData) {
        if (vectorData instanceof Float32Array) return vectorData;
        // Buffer/Uint8Array from SQLite: 需要按字节拷贝重新解释
        const result = new Float32Array(this.config.dimension);
        new Uint8Array(result.buffer).set(vectorData);
        return result;
    }

    _magnitude(vec) {
        let sum = 0;
        for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
        return Math.sqrt(sum);
    }

    _dotProduct(v1, v2) {
        let sum = 0;
        for (let i = 0; i < v1.length; i++) sum += v1[i] * v2[i];
        return sum;
    }

    _emptyResult(dim) {
        return {
            levels: [],
            totalExplainedEnergy: 0,
            finalResidual: new Float32Array(dim),
            features: { depth: 0, coverage: 0, novelty: 1, coherence: 0, tagMemoActivation: 0 }
        };
    }
}