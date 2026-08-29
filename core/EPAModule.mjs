/**
 * core/EPAModule.mjs — 移植文件(SPEC-P1.md §1)
 * 出处:VCPToolBox (https://github.com/lioensky/VCPToolBox) — EPAModule.js(CommonJS,741 行)
 * 授权:CC BY-NC-SA 4.0(署名 lioensky/VCPToolBox)
 * 移植日期:2026-08-29
 *
 * 移植规则(SPEC-P1.md §1,数学代码逐行保留,未重写):
 * - 仅替换外部接口:构造参数 (db, config) → (tagProvider, cache, config);
 * - vector 由 SQLite Buffer 改为 Float32Array 入参(_clusterTags 读取处统一适配);
 * - 缓存 key 沿用 'epa_basis_cache';缓存值 JSON 可序列化(basis/mean 转普通数组);
 * - 原默认配置 dim=3072/maxBasisDim=64/clusterCount=64 → dimension=1024/maxBasisDim=32/clusterCount=12,
 *   minTags 由硬编码 8 改为配置项(默认 8);
 * - 删除 Rust/vexus 相关分支与 refreshInBackground(详见移植汇报):
 *   config.vexusIndex / deferRustRecompute / afterRustWrite / withRustWriteLease、
 *   initialize() 的 Rust 训练分支、project() 的 Rust 投影快路径、
 *   _shouldLogRustSummary / _logRustEpaSummary / _recomputeWithRust / refreshInBackground /
 *   _loadBoundedTagSnapshot / _computeBasisFromSnapshot / _publishBasisCacheWithLease /
 *   _refreshFlattenedBasisCache / _getFlattenedBasis 与 _flattenedBasisCache 字段;
 * - 原代码含两处 Math.random(均按规格原样保留):① _clusterTags 的 K-Means Forgy 质心选取,
 *   ② _powerIteration 的幂迭代起始向量。
 */

/**
 * EPAModule.js (Physics-Optimized Edition)
 * 嵌入投影分析模块
 * 优化点:加权中心化 PCA、鲁棒 K-Means、基于能量共现的共振检测
 */

export class EPAModule {
    constructor(tagProvider, cache, config = {}) {
        // 外部接口(SPEC-P1.md §1):
        // tagProvider.listTagVectors() → Array<{ id:number, name:string, vector:Float32Array }>(有向量的记录)
        // cache.get(key)/cache.set(key, value) → 同步 KV,持久化由调用方负责
        this.tagProvider = tagProvider;
        this.cache = cache;
        this.config = {
            maxBasisDim: config.maxBasisDim || 32,
            minVarianceRatio: config.minVarianceRatio || 0.01,
            clusterCount: config.clusterCount || 12,
            dimension: config.dimension || 1024,
            minTags: config.minTags || 8,
            strictOrthogonalization: config.strictOrthogonalization !== undefined ? config.strictOrthogonalization : true,
            ...config
        };

        this.orthoBasis = null;      // 正交基向量 (Float32Array[])
        this.basisMean = null;       // 🌟 新增:全局加权平均向量 (用于中心化)
        this.basisLabels = null;     // 基底标签
        this.basisEnergies = null;   // 特征值 (方差贡献)

        this.initialized = false;
    }

    /** 训练状态:缓存命中加载或训练成功后为 true */
    get trained() {
        return this.initialized;
    }

    async initialize() {
        console.log('[EPA] 🧠 Initializing orthogonal basis (Weighted PCA)...');

        try {
            if (await this._loadFromCache()) {
                console.log(`[EPA] 💾 Loaded basis from cache.`);
                this.initialized = true;
                return true;
            }

            // 原代码:this.db.prepare(`SELECT id, name, vector FROM tags WHERE vector IS NOT NULL`).all()
            // 移植:由 tagProvider 直接提供有向量的 Float32Array 记录(SQLite Buffer → Float32Array 在读取处适配)
            const tags = this.tagProvider.listTagVectors();
            if (tags.length < this.config.minTags) return false;

            // 1. 鲁棒 K-Means 聚类 (提取加权质心)
            const clusterData = this._clusterTags(tags, Math.min(tags.length, this.config.clusterCount));

            // 2. 🌟 计算 SVD (加权中心化 PCA)
            // 相比之前的纯 SVD,这里先去中心化,再加权,更能提取差异特征
            const svdResult = this._computeWeightedPCA(clusterData);

            const { U, S, meanVector, labels } = svdResult;

            // 3. 选择主成分
            const K = this._selectBasisDimension(S);

            this.orthoBasis = U.slice(0, K);
            this.basisEnergies = S.slice(0, K);
            this.basisMean = meanVector; // 保存平均向量用于投影时的去中心化
            this.basisLabels = labels ? labels.slice(0, K) : clusterData.labels.slice(0, K);

            await this._saveToCache();

            this.initialized = true;
            return true;
        } catch (e) {
            console.error('[EPA] ❌ Init failed:', e);
            return false;
        }
    }

    /**
     * 投影向量到语义空间
     * ⚠️ 修正:必须先减去平均向量 (Centering),否则投影没有统计意义
     */
    project(vector) {
        // 契约(SPEC-P1.md §1):未训练时返回 null(原实现此处返回 _emptyResult)
        if (!this.initialized || !this.orthoBasis) return null;

        const vec = vector instanceof Float32Array ? vector : new Float32Array(vector);
        const dim = vec.length;
        const K = this.orthoBasis.length;

        let projections, probabilities, entropy, totalEnergy;

        // 原代码此处有 Rust 快路径(vexusIndex.project),已删除;此判断恒真,仅保留原结构
        if (!projections) {
            // 1. 去中心化: v' = v - mean
            const centeredVec = new Float32Array(dim);
            for (let i = 0; i < dim; i++) centeredVec[i] = vec[i] - this.basisMean[i];

            projections = new Float32Array(K);
            totalEnergy = 0;

            // 2. 投影到主成分轴
            for (let k = 0; k < K; k++) {
                let dot = 0;
                const basis = this.orthoBasis[k];
                for (let d = 0; d < dim; d++) {
                    dot += centeredVec[d] * basis[d];
                }
                projections[k] = dot;
                totalEnergy += dot * dot;
            }

            if (totalEnergy < 1e-12) return this._emptyResult();

            // 3. 计算熵 (信息散度)
            probabilities = new Float32Array(K);
            entropy = 0;
            for (let k = 0; k < K; k++) {
                probabilities[k] = (projections[k] * projections[k]) / totalEnergy;
                if (probabilities[k] > 1e-9) {
                    entropy -= probabilities[k] * Math.log2(probabilities[k]);
                }
            }
        }

        const normalizedEntropy = K > 1 ? entropy / Math.log2(K) : 0;

        // 4. 提取主轴
        const dominantAxes = [];
        for (let k = 0; k < K; k++) {
            // 阈值下调,因为去中心化后能量更分散
            if (probabilities[k] > 0.05) {
                dominantAxes.push({
                    index: k,
                    label: this.basisLabels[k],
                    energy: probabilities[k],
                    projection: projections[k] // 保留正负号,表示在轴上的方向
                });
            }
        }
        dominantAxes.sort((a, b) => b.energy - a.energy);

        return {
            projections,
            probabilities,
            entropy: normalizedEntropy,
            logicDepth: 1 - normalizedEntropy, // 熵低则逻辑深度高 (聚焦)
            dominantAxes
        };
    }

    /**
     * 🌟 修正:跨域共振检测
     * 逻辑:检测是否"同时"强激活了两个"正交"的语义轴。
     * 因为基底本身已经是正交的,所以不需要计算基底相似度。
     * 我们计算的是 Query 在这些互斥轴上的共现强度 (Co-occurrence Power)。
     */
    detectCrossDomainResonance(vector) {
        // 契约(SPEC-P1.md §1):未训练时返回 null(project 未训练返回 null,先守卫避免解构报错)
        if (!this.initialized || !this.orthoBasis) return null;

        const { dominantAxes } = this.project(vector);
        if (dominantAxes.length < 2) return { resonance: 0, bridges: [] };

        const bridges = [];
        const topAxis = dominantAxes[0];

        // 只检查与最强轴共振的其他轴
        for (let i = 1; i < dominantAxes.length; i++) {
            const secondaryAxis = dominantAxes[i];

            // 几何平均能量: sqrt(E1 * E2)
            // 这代表两个轴同时被激活的程度。如果一个极强一个极弱,乘积会很小。
            const coActivation = Math.sqrt(topAxis.energy * secondaryAxis.energy);

            // 只有当共激活强度足够大时,才视为"共振"
            if (coActivation > 0.15) {
                bridges.push({
                    from: topAxis.label,
                    to: secondaryAxis.label,
                    strength: coActivation,
                    // Distance 在这里是隐喻,因为轴是正交的,距离恒定。
                    // 我们可以用能量比率来表示"平衡度"
                    balance: Math.min(topAxis.energy, secondaryAxis.energy) / Math.max(topAxis.energy, secondaryAxis.energy)
                });
            }
        }

        // 总共振值 = 所有 Bridge 强度的总和
        const resonance = bridges.reduce((sum, b) => sum + b.strength, 0);
        return { resonance, bridges };
    }

    // --- 数学核心优化 ---

    /**
     * 🌟 优化:带收敛检测和权重的 K-Means
     */
    _clusterTags(tags, k) {
        const startedAt = Date.now();
        const dim = this.config.dimension;
        const vectors = tags.map(t => {
            // 原代码:SQLite Buffer 字节拷贝进入同长 Float32Array;
            // 移植:入参 vector 已是 Float32Array,改为等长拷贝(超长抛错/不足补零行为与原字节拷贝一致)
            const aligned = new Float32Array(dim);
            aligned.set(t.vector);
            return aligned;
        });

        // 初始化:随机选择 k 个点作为初始质心 (Forgy Method)
        // ⚠️ 此处 Math.random 为原代码保留(SPEC-P1.md §1:保持原样不改)
        let centroids = [];
        const indices = new Set();
        while (indices.size < k) indices.add(Math.floor(Math.random() * vectors.length));
        centroids = Array.from(indices).map(i => new Float32Array(vectors[i]));

        let clusterSizes = new Float32Array(k);
        const maxIter = 50; // 增加迭代次数
        const tolerance = 1e-4; // 收敛阈值

        for (let iter = 0; iter < maxIter; iter++) {
            if (iter === 0 || (iter + 1) % 10 === 0) {
                console.log(`[EPA] 🧮 JS K-Means progress: iter=${iter + 1}/${maxIter}, tags=${vectors.length}, clusters=${k}, elapsed=${Date.now() - startedAt}ms`);
            }
            const clusters = Array.from({ length: k }, () => []);
            let movement = 0;

            // Assign
            vectors.forEach(v => {
                let maxSim = -Infinity, bestK = 0;
                // 优化:使用点积代替距离(假设向量已归一化),速度更快
                centroids.forEach((c, i) => {
                    let dot = 0;
                    for (let d = 0; d < dim; d++) dot += v[d] * c[d];
                    if (dot > maxSim) { maxSim = dot; bestK = i; }
                });
                clusters[bestK].push(v);
            });

            // Update
            const newCentroids = clusters.map((cvs, i) => {
                if (cvs.length === 0) return centroids[i]; // 避免空簇
                const newC = new Float32Array(dim);
                cvs.forEach(v => { for (let d = 0; d < dim; d++) newC[d] += v[d]; });

                // 归一化新质心
                let mag = 0;
                for (let d = 0; d < dim; d++) mag += newC[d] ** 2;
                mag = Math.sqrt(mag);
                if (mag > 1e-9) for (let d = 0; d < dim; d++) newC[d] /= mag;

                // 计算移动距离 (Euclidean check for convergence)
                let distSq = 0;
                for (let d = 0; d < dim; d++) distSq += (newC[d] - centroids[i][d]) ** 2;
                movement += distSq;

                return newC;
            });

            clusterSizes = clusters.map(c => c.length);
            centroids = newCentroids;

            if (movement < tolerance) {
                // console.log(`[EPA] K-Means converged at iter ${iter}`);
                break;
            }
        }

        console.log(`[EPA] 🧮 JS K-Means assignment complete: elapsed=${Date.now() - startedAt}ms`);

        // 命名逻辑不变
        const labels = centroids.map(c => {
            let maxSim = -Infinity, closest = 'Unknown';
            vectors.forEach((v, i) => {
                let dot = 0;
                for (let d = 0; d < dim; d++) dot += c[d] * v[d];
                if (dot > maxSim) { maxSim = dot; closest = tags[i].name; }
            });
            return closest;
        });

        // 🌟 返回 weights (簇大小),这对于 PCA 很重要
        return { vectors: centroids, labels, weights: clusterSizes };
    }

    /**
     * 🌟 核心算法:加权 PCA (基于 SVD)
     * 步骤:
     * 1. 计算加权平均值 (Weighted Mean)
     * 2. 中心化矩阵 (Centering)
     * 3. 构建加权协方差矩阵的近似 (Weighted Gram Matrix)
     * 4. Power Iteration 提取特征向量
     */
    _computeWeightedPCA(clusterData) {
        const startedAt = Date.now();
        const { vectors, weights } = clusterData;
        const n = vectors.length;
        const dim = this.config.dimension;
        console.log(`[EPA] 🧮 JS weighted PCA started: clusters=${n}, dim=${dim}, maxBasis=${this.config.maxBasisDim}`);
        const totalWeight = weights.reduce((a, b) => a + b, 0);

        // 1. 计算全局加权平均向量
        const meanVector = new Float32Array(dim);
        for (let i = 0; i < n; i++) {
            const w = weights[i];
            for (let d = 0; d < dim; d++) {
                meanVector[d] += vectors[i][d] * w;
            }
        }
        for (let d = 0; d < dim; d++) meanVector[d] /= totalWeight;

        // 2. 隐式构建加权 Gram 矩阵 (G = X_centered * W * X_centered^T)
        // 我们不需要显式构建 huge covariance matrix (dim*dim),而是构建 Gram matrix (n*n)
        // 这里的 X_centered 行向量其实是: sqrt(w_i) * (v_i - mean)

        const centeredScaledVectors = vectors.map((v, i) => {
            const vec = new Float32Array(dim);
            const scale = Math.sqrt(weights[i]); // 权重的平方根
            for (let d = 0; d < dim; d++) {
                vec[d] = (v[d] - meanVector[d]) * scale;
            }
            return vec;
        });

        // Gram Matrix (n x n)
        const gram = new Float32Array(n * n);
        for (let i = 0; i < n; i++) {
            for (let j = i; j < n; j++) {
                let dot = 0;
                // SIMD 优化点
                for (let d = 0; d < dim; d++) dot += centeredScaledVectors[i][d] * centeredScaledVectors[j][d];
                gram[i * n + j] = gram[j * n + i] = dot;
            }
        }

        // 3. Power Iteration with Re-orthogonalization
        const eigenvectors = []; // U
        const eigenvalues = [];  // S
        const gramCopy = new Float32Array(gram);

        const maxBasis = Math.min(n, this.config.maxBasisDim);

        for (let k = 0; k < maxBasis; k++) {
            if (k === 0 || (k + 1) % 8 === 0) {
                console.log(`[EPA] 🧮 JS weighted PCA progress: basis=${k + 1}/${maxBasis}, elapsed=${Date.now() - startedAt}ms`);
            }
            const { vector: v, value } = this._powerIteration(gramCopy, n, eigenvectors);
            if (value < 1e-6) break; // 特征值太小

            eigenvectors.push(v);
            eigenvalues.push(value);

            // Deflation: G_new = G_old - lambda * v * v^T
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    gramCopy[i * n + j] -= value * v[i] * v[j];
                }
            }
        }

        // 4. 将 Gram 矩阵的特征向量 v (维度 n) 映射回原始空间 (维度 dim)
        // U_pca = X^T * v / sqrt(lambda)
        const U = eigenvectors.map((ev, idx) => {
            const lambda = eigenvalues[idx];
            const basis = new Float32Array(dim);

            // 线性组合
            for (let i = 0; i < n; i++) {
                const weight = ev[i]; // Gram 特征向量的分量
                if (Math.abs(weight) > 1e-9) {
                    for (let d = 0; d < dim; d++) {
                        basis[d] += weight * centeredScaledVectors[i][d];
                    }
                }
            }

            // 归一化
            let mag = 0;
            for (let d = 0; d < dim; d++) mag += basis[d] ** 2;
            mag = Math.sqrt(mag);
            if (mag > 1e-9) for (let d = 0; d < dim; d++) basis[d] /= mag;

            return basis;
        });

        console.log(`[EPA] 🧮 JS weighted PCA finished: basis=${U.length}, elapsed=${Date.now() - startedAt}ms`);
        return { U, S: eigenvalues, meanVector, labels: clusterData.labels };
    }

    _powerIteration(matrix, n, existingBasis) {
        // 随机初始化
        // ⚠️ 此处 Math.random 为原代码保留(SPEC-P1.md §1:数学实现逐行保留)
        let v = new Float32Array(n).map(() => Math.random() - 0.5);
        let lastVal = 0;

        for (let iter = 0; iter < 100; iter++) {
            const w = new Float32Array(n);

            // Matrix-Vector Multiplication
            for (let r = 0; r < n; r++) {
                for (let c = 0; c < n; c++) w[r] += matrix[r * n + c] * v[c];
            }

            // 🌟 关键优化:Re-orthogonalization (Gram-Schmidt against existing)
            // 防止幂迭代收敛到已经找到的主成分上(解决 Deflation 精度丢失问题)
            // 注意:因为我们是对 Gram 矩阵做分解,这里的 existingBasis 是 n 维向量。
            if (this.config.strictOrthogonalization && existingBasis && existingBasis.length > 0) {
                for (const prevV of existingBasis) {
                    let dot = 0;
                    for (let i = 0; i < n; i++) dot += w[i] * prevV[i];
                    for (let i = 0; i < n; i++) w[i] -= dot * prevV[i];
                }
            }

            // Rayleigh Quotient (在正交化之后计算,避免混入已提取主成分的能量)
            let val = 0;
            for (let i = 0; i < n; i++) val += v[i] * w[i];

            // Normalize
            let mag = 0;
            for (let i = 0; i < n; i++) mag += w[i] ** 2;
            mag = Math.sqrt(mag);

            if (mag < 1e-9) break;

            for (let i = 0; i < n; i++) v[i] = w[i] / mag;

            if (Math.abs(val - lastVal) < 1e-6) {
                lastVal = val;
                break;
            }
            lastVal = val;
        }
        return { vector: v, value: lastVal };
    }

    _selectBasisDimension(S) {
        const total = S.reduce((a, b) => a + b, 0);
        let cum = 0;
        // 稍微提高解释方差比例 (0.9 -> 0.95),因为PCA后数据更集中
        for (let i = 0; i < S.length; i++) {
            cum += S[i];
            if (cum / total > 0.95) return Math.max(i + 1, 8);
        }
        return S.length;
    }

    async _saveToCache() {
        try {
            // 缓存值必须 JSON 可序列化(SPEC-P1.md §1):basis/mean 由 Float32Array 转普通数组
            // (原代码用 Buffer base64;普通数组经 JSON 往返无损——float32 值在 float64 中精确可表示)
            const data = {
                basis: this.orthoBasis.map(b => Array.from(b)),
                mean: Array.from(this.basisMean), // 🌟 Save Mean
                energies: Array.from(this.basisEnergies),
                labels: this.basisLabels,
                timestamp: Date.now(),
                // 原代码统计 db 全部 tag 数;移植后 tagProvider 只有有向量的记录,以其长度代替(仅信息字段)
                tagCount: this.tagProvider.listTagVectors().length
            };
            this.cache.set('epa_basis_cache', data);
        } catch (e) { console.error('[EPA] Save cache error:', e); }
    }

    async _loadFromCache(options = {}) {
        try {
            const raw = this.cache.get('epa_basis_cache');
            if (!raw) return false;
            // 兼容调用方持久化层以 JSON 字符串形式存储
            const data = typeof raw === 'string' ? JSON.parse(raw) : raw;

            // 简单校验
            if (!data.mean) return false; // 旧缓存格式不兼容
            if (options.expectedTagCount && data.tagCount && data.tagCount !== options.expectedTagCount) return false;

            this.orthoBasis = data.basis.map(arr => {
                // 原代码:base64 → Buffer → Float32Array;移植:普通数组 → Float32Array
                const aligned = Float32Array.from(arr);
                return aligned;
            });
            this.basisMean = Float32Array.from(data.mean);
            this.basisEnergies = new Float32Array(data.energies);
            this.basisLabels = data.labels;
            return true;
        } catch (e) { return false; }
    }

    _emptyResult() {
        return { projections: null, probabilities: null, entropy: 1, logicDepth: 0, dominantAxes: [] };
    }
}