# Multi-Stage Historical Document Digitization Pipeline
## Project Development Plan & Timeline

**Student:** Mohammed Elhasnaoui #300268139  
**Project Duration:** ~12-14 weeks  
**Last Updated:** January 27, 2026

---

## Executive Summary

This project builds an intelligent, cost-efficient OCR pipeline for historical documents that:
1. Preprocesses and enhances degraded document images
2. Classifies documents by difficulty
3. Routes them through appropriate OCR engines (Tesseract, custom TensorFlow model, heavy pre-trained models)
4. Achieves 60-80% cost reduction compared to all-cloud processing

---

## Goals Definition

### Main Goals (Must Complete)

1. **Custom TensorFlow OCR Model**: Train a handwriting recognition model from scratch using CNN-LSTM-CTC architecture
2. **3-Tier OCR Routing System**: Implement easy/medium/hard classification and routing
3. **Basic Preprocessing Pipeline**: Deskewing, binarization, denoising using OpenCV
4. **Benchmark Demonstration**: Show cost savings and accuracy across multiple test sets
5. **Working End-to-End System**: Accept document images, output transcribed text

### Stretch Goals (If Time Permits)

1. **Active Learning System**: Automatically identify and prioritize documents for human correction/model retraining
2. **Web Service Deployment**: Deploy as accessible API with web interface
3. **Direct API Cost Comparison**: Quantitative comparison against Google Vision API and AWS Textract
4. **Fine-tuning on Historical Collections**: Adapt model to specific archive styles (e.g., 19th-century medical records)
5. **Advanced Transformer Model**: Replace CNN-LSTM-CTC with Transformer-based architecture

**Decision Point:** Week 9 - Assess main goal completion and decide which stretch goals to pursue

---

## Development Phases

### Phase 0: Setup & Preparation (Week 1)
**Duration:** 5-7 days  
**Goal:** Get development environment ready and understand existing tools

**Tasks:**
- [ ] Set up Python environment with TensorFlow, OpenCV, Tesseract
- [ ] Configure GPU/CUDA if available, otherwise prepare for CPU training
- [ ] Download and organize datasets (NIST, IAM, EMNIST)
- [ ] Explore existing OCR tools (Tesseract, TrOCR) with sample documents
- [ ] Set up version control (Git) and experiment tracking (TensorBoard, MLflow)
- [ ] Create basic project structure and documentation

**Deliverables:**
- Working development environment
- Dataset download scripts
- Initial project repository
- Development environment documentation

**Risks:**
- Dataset access issues (IAM requires registration)
- GPU setup complications
- Large dataset download times

---

### Phase 1: Preprocessing Pipeline (Week 2-3)
**Duration:** 10-12 days  
**Goal:** Build robust document image enhancement system

**Tasks:**
- [ ] Implement deskewing (rotation correction) using OpenCV
- [ ] Implement adaptive binarization for faded ink (Otsu, Sauvola methods)
- [ ] Implement denoising (Gaussian blur, bilateral filter, morphological operations)
- [ ] Implement contrast enhancement (CLAHE, histogram equalization)
- [ ] Create preprocessing configuration system (different profiles for different document types)
- [ ] Test on sample degraded documents and evaluate visually
- [ ] Build preprocessing benchmark (before/after image quality metrics)

**Deliverables:**
- Preprocessing module with configurable pipeline
- Before/after visualization tool
- Performance benchmarks (processing time per image)

**Success Criteria:**
- Process 1000x1500px image in <500ms
- Visibly improved text clarity on degraded samples

---

### Phase 2: Difficulty Classifier (Week 3-4)
**Duration:** 8-10 days  
**Goal:** Build lightweight classifier to route documents to appropriate OCR engines

**Tasks:**
- [ ] Create labeled difficulty dataset (easy/medium/hard)
  - Easy: Clean printed documents
  - Medium: Clear handwriting, forms, isolated characters
  - Hard: Cursive, severely degraded, damaged pages
- [ ] Design lightweight CNN classifier (MobileNet-based or custom small CNN)
- [ ] Train classifier with data augmentation
- [ ] Evaluate classifier accuracy (target: >85%)
- [ ] Optimize for speed (target: <10ms per image)
- [ ] Implement routing logic based on classifier output

**Deliverables:**
- Trained difficulty classifier model
- Routing module with configurable thresholds
- Classification accuracy report

**Success Criteria:**
- >85% classification accuracy on test set
- <10ms inference time per document

**Risks:**
- Creating balanced difficulty dataset may be time-consuming
- Subjective definition of "difficulty" may require iteration

---

### Phase 3: Custom TensorFlow OCR Model (Week 4-8)
**Duration:** 4 weeks (CORE COMPONENT)  
**Goal:** Design, train, and optimize handwriting recognition model from scratch

#### Week 4-5: Architecture Design & Initial Training
**Tasks:**
- [ ] Research and select architecture (CNN-LSTM-CTC recommended to start)
  - Convolutional layers for feature extraction
  - Bidirectional LSTM for sequence modeling
  - CTC loss for alignment-free training
- [ ] Implement model architecture in TensorFlow/Keras
- [ ] Build efficient data pipeline using tf.data
- [ ] Implement data augmentation (rotation, noise, blur, elastic distortions)
- [ ] Configure CTC loss and decoder
- [ ] Run initial training on small subset to validate pipeline

**Deliverables:**
- Model architecture code
- Training pipeline with tf.data
- Initial training run results

#### Week 6-7: Full Training & Hyperparameter Tuning
**Tasks:**
- [ ] Train on full NIST + IAM + EMNIST datasets
- [ ] Implement learning rate scheduling
- [ ] Experiment with different architectures (layer depths, LSTM units)
- [ ] Monitor training with TensorBoard (loss curves, sample predictions)
- [ ] Implement early stopping and checkpointing
- [ ] Run hyperparameter search (learning rate, batch size, augmentation strength)

**Deliverables:**
- Fully trained model weights
- Training logs and TensorBoard visualizations
- Hyperparameter search results

**Success Criteria:**
- CER < 15% on IAM test set (medium difficulty handwriting)
- CER < 10% on NIST test set (handprinted characters)

#### Week 8: Model Optimization & Validation
**Tasks:**
- [ ] Evaluate model on held-out test sets
- [ ] Analyze error patterns (which characters/words fail)
- [ ] Implement TensorFlow Lite quantization (FP32 → INT8)
- [ ] Benchmark inference speed (quantized vs. full precision)
- [ ] Test on real historical documents (not from training set)
- [ ] Document model architecture and training process

**Deliverables:**
- Quantized TFLite model
- Comprehensive evaluation report (CER, WER by dataset)
- Error analysis document
- Inference speed benchmarks

**Risks:**
- Training time may be longer than expected (adjust batch size, use mixed precision)
- Initial model may underperform (need architecture iteration)
- Overfitting on training data (monitor validation curves closely)

---

### Phase 4: Integration & Pipeline Assembly (Week 9-10)
**Duration:** 10-12 days  
**Goal:** Combine all components into working end-to-end system

**Tasks:**
- [ ] Integrate Tesseract OCR for easy documents
- [ ] Integrate heavy pre-trained model (TrOCR-large or PaddleOCR) for hard documents
- [ ] Build pipeline orchestration (preprocessing → classification → routing → OCR)
- [ ] Implement post-processing (language model correction, confidence scoring)
- [ ] Create batch processing system for multiple documents
- [ ] Build output generation (searchable PDF, plain text, JSON with confidence scores)
- [ ] Implement logging and error handling

**Deliverables:**
- Complete pipeline code (FastAPI or Flask application)
- API endpoints for document submission and result retrieval
- Batch processing scripts
- Integration test suite

**Success Criteria:**
- Process 100 mixed-difficulty documents without errors
- Correctly route documents to appropriate OCR engines
- Generate outputs in multiple formats

---

### Phase 5: Comprehensive Benchmarking (Week 11-12)
**Duration:** 10-12 days  
**Goal:** Demonstrate cost-efficiency and accuracy gains

**Tasks:**
- [ ] Define baseline comparisons:
  - All-Tesseract (cheapest but lowest accuracy)
  - All-TrOCR (most accurate but most expensive)
  - All-custom model (single engine)
  - Smart routing (our approach)
- [ ] Collect diverse test set (500-1000 documents, mixed difficulty)
- [ ] Run all baselines and measure:
  - Accuracy (CER, WER by difficulty tier)
  - Cost (compute time, API calls, GPU-hours)
  - Processing speed (documents per minute)
- [ ] Generate comparison charts and tables
- [ ] Calculate cost savings percentage
- [ ] Document methodology for reproducibility

**Deliverables:**
- Benchmark report with charts and statistical analysis
- Test dataset with ground truth annotations
- Benchmark scripts for reproducibility

**Success Criteria:**
- Demonstrate 60-80% cost reduction vs. all-TrOCR baseline
- Maintain accuracy within 5% of all-TrOCR on hard documents
- Show faster processing than all-cloud approach

---

### Phase 6: Dashboard & Visualization (Week 12-13)
**Duration:** 7-10 days  
**Goal:** Build interactive dashboard for results visualization

**Tasks:**
- [ ] Design dashboard layout (documents processed, routing stats, accuracy by tier)
- [ ] Implement real-time processing monitor
- [ ] Integrate TensorBoard training curves
- [ ] Create live inference demo (upload document → see routing → view results)
- [ ] Add cost calculator (estimate savings for different document volumes)
- [ ] Build comparison visualizations (accuracy vs. cost tradeoffs)

**Deliverables:**
- Interactive web dashboard (Streamlit, Dash, or custom React app)
- Live demo system
- Cost calculator tool

---

### Phase 7: Documentation & Polish (Week 13-14)
**Duration:** 7-10 days  
**Goal:** Complete documentation and prepare final deliverables

**Tasks:**
- [ ] Write comprehensive README with quick-start guide
- [ ] Document API endpoints and usage examples
- [ ] Create architecture diagrams and flowcharts
- [ ] Write technical report on custom model training
- [ ] Record demo video showing pipeline in action
- [ ] Clean up code, add comments, ensure PEP 8 compliance
- [ ] Prepare final presentation materials

**Deliverables:**
- Complete documentation suite
- Demo video
- Final presentation slides
- Clean, commented codebase

---

## Milestones & Checkpoints

| Week | Milestone | Success Criteria |
|------|-----------|------------------|
| 1 | Environment Ready | Can run Tesseract and TensorFlow hello-world |
| 3 | Preprocessing + Classifier Working | Can enhance images and classify difficulty |
| 5 | Model Training Launched | CNN-LSTM-CTC training with decreasing loss |
| 8 | Custom Model Complete | CER < 15% on IAM test set |
| 10 | Full Pipeline Integrated | End-to-end processing of test documents |
| 12 | Benchmarks Complete | Cost savings demonstrated with data |
| 14 | Project Complete | All deliverables ready for submission |

**Decision Point (Week 9):** If main goals are on track, begin stretch goals. If behind schedule, focus only on core deliverables.

---

## Dependencies & Critical Path

**Critical Path:**
1. Environment Setup (blocks everything)
2. Dataset Preparation (blocks model training)
3. Custom Model Training (longest task, blocks integration)
4. Pipeline Integration (blocks benchmarking)
5. Benchmarking (blocks final report)

**Parallel Work Opportunities:**
- Preprocessing pipeline can be developed while datasets download
- Difficulty classifier can be trained while custom OCR model trains
- Dashboard design can happen while benchmarks run

---

## Risk Management

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Model training takes too long | Medium | High | Use mixed precision training, smaller batch sizes, cloud GPU if needed |
| IAM dataset access denied | Low | Medium | Use only NIST + EMNIST, or synthetic handwriting data |
| Custom model underperforms | Medium | High | Start simple (CNN-LSTM-CTC), iterate quickly, have fallback to more pre-trained models |
| Integration complexity | Medium | Medium | Build modular components with clear interfaces |
| Insufficient GPU resources | Medium | Medium | Use Google Colab, AWS free tier, or train smaller model |

---

## Success Metrics (Final Evaluation)

### Technical Metrics
- [ ] Custom OCR model CER < 15% on handwriting test sets
- [ ] Difficulty classifier accuracy > 85%
- [ ] 60-80% cost reduction vs. all-cloud baseline
- [ ] End-to-end pipeline processes 100+ documents successfully

### Deliverable Completeness
- [ ] All 5 main deliverables completed and documented
- [ ] Code is clean, modular, and well-commented
- [ ] Benchmark report with statistical analysis
- [ ] Working dashboard with live demo

### Demonstration Quality
- [ ] Can process real historical documents not in training set
- [ ] Dashboard effectively visualizes results
- [ ] Technical report clearly explains methodology

---

## Weekly Time Allocation

**Recommended Weekly Hours:** 15-20 hours

**Breakdown:**
- Implementation: 10-12 hours
- Reading/Research: 2-3 hours
- Documentation: 2-3 hours
- Experimentation/Debugging: 3-5 hours

**Total Project Hours:** ~200-250 hours

---

## Next Steps

1. Review and approve this plan
2. Set up development environment (Week 1)
3. Create GitHub repository and project structure
4. Begin Phase 0 tasks immediately

**First Action Item:** Run `pip install tensorflow opencv-python pytesseract` and verify installations.