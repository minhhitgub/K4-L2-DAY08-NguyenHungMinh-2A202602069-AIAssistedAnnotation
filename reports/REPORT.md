# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Hùng Minh

Công cụ gán nhãn đã dùng: CVAT (CVAT Docker local v2.76.0, định dạng Ultralytics YOLO Detection 1.0)

Báo cáo này phân tích chi tiết quy trình học chủ động (Active Learning) trên bài toán phát hiện xe hơi ban đêm từ video camera giám sát giao thông. Mọi số liệu trong báo cáo đều được đối chiếu và truy xuất trực tiếp từ các file số đo trong hệ thống: `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`, `outputs/round1_diff.json` và `outputs/round1_diff.md`.

---

## 1. Dữ liệu và cách chia tập

Dữ liệu thực nghiệm được trích xuất từ video camera giám sát giao thông ban đêm (`Ina7KMV2OEI`) quay liên tục từ một vị trí cố định trên cầu vượt nhìn xuống đường cao tốc, lấy mẫu ở tốc độ 2.5 khung hình/giây (fps) với độ phân giải 1280×720, gồm 400 khung hình từ `frame_0000.jpg` đến `frame_0399.jpg`.

Tập dữ liệu được chia theo trục thời gian với vùng đệm bảo vệ (temporal split with buffer zones):
- **Tập kiểm thử (test set):** Gồm 20 ảnh lấy từ 4 đoạn cố định có tâm tại giây 20, 60, 100 và 140 (mỗi đoạn lấy 5 ảnh cách nhau 1.2 giây).
- **Vùng đệm (buffer zones):** Gồm 112 ảnh trong khoảng 4 giây trước và sau mỗi đoạn kiểm thử, cùng các ảnh nằm xen kẽ giữa các ảnh kiểm thử, hoàn toàn bị loại bỏ khỏi cả tập train lẫn test.
- **Tập chưa gán nhãn (pool):** Gồm 268 ảnh còn lại; mọi vòng học chủ động chỉ được phép chọn ảnh huấn luyện từ tập này. Nhờ vùng đệm, ảnh pool gần ảnh test nhất vẫn cách nhau tối thiểu 4.4 giây.

**Lý do không chia ngẫu nhiên (random split):**
Vì camera được đặt cố định, hai khung hình liên tiếp cách nhau 0.4 giây có hậu cảnh (đường sá, cầu vượt, dải phân cách, đèn đường) gần như giống hệt nhau. Hơn nữa, mỗi chiếc xe di chuyển qua khung hình mất từ vài giây đến hơn chục giây. Nếu chia ngẫu nhiên, cùng một chiếc xe (với cùng hướng di chuyển, màu sắc, góc phản chiếu ánh đèn) sẽ đồng thời xuất hiện ở cả tập huấn luyện và tập kiểm thử. 

**Hướng lệch của số đo nếu chia ngẫu nhiên:**
Nếu chia ngẫu nhiên, hiện tượng **rò rỉ dữ liệu (data leakage)** nghiêm trọng sẽ xảy ra. Số đo trên tập kiểm thử (đặc biệt là AP50, Precision và Recall) sẽ bị **lệch cao hơn thực tế (over-optimistic)** rất nhiều. Mô hình khi đó không chứng minh được khả năng khái quát hóa (generalization) trên các xe mới hay các thời điểm mới, mà chỉ đơn thuần "học vẹt" (ghi nhớ) hình ảnh của những chiếc xe cụ thể mà nó đã nhìn thấy trong lúc huấn luyện. Việc chia theo trục thời gian và cách ly bằng vùng đệm 4.4 giây là bắt buộc để đánh giá đúng năng lực thực tế của mô hình.

---

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng số đo vòng 0 từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào `outputs/metrics_round0.json` và trực quan hóa trong `outputs/compare_round0.jpg`:
- **Độ chính xác và độ phủ tổng thể:** Mô hình ban đầu đạt độ chính xác (Precision@0.25) rất cao là **0.925** (92.5%, TP=197, FP=16), chứng tỏ khi mô hình phát hiện ra xe thì độ tin cậy khá tốt, ít bắt nhầm vật thể lạ. Tuy nhiên, độ phủ (Recall@0.25) chỉ đạt **0.489** (48.9%), bỏ sót tới **206 box** trên tổng số 403 box tham chiếu (FN=206).
- **Phân tích Recall theo kích thước xe (`recall_by_size`):**
  - **Small (xe nhỏ, ở xa):** Recall chỉ đạt **0.1818** (18.2%, bắt được 12/66 box). Mô hình gần như bỏ sót hoàn toàn các xe ở rất xa gần đường chân trời, nơi xe chỉ thu lại thành hai chấm đèn pha hoặc đèn hậu mờ nhạt.
  - **Medium (xe tầm trung):** Recall đạt **0.5473** (54.7%, bắt được 162/296 box). Bỏ sót gần một nửa số xe, đặc biệt là các xe đi ở làn tối, xe ngược chiều bị lóa đèn, hoặc xe màu đen hòa vào bóng tối.
  - **Large (xe lớn, gần camera):** Recall đạt **0.5610** (56.1%, bắt được 23/41 box). Dù ở gần nhưng mô hình vẫn bỏ sót các xe tải lớn có thùng xe tối, hoặc xe bị cắt một phần thân ở rìa dưới khung hình do hình dạng khác biệt với dữ liệu COCO ban ngày.
- **Những loại xe không khớp nhãn tham chiếu:** 
  1. Xe chạy sát mép ảnh (bị cắt nửa thân).
  2. Xe ở các làn đường xa (chỉ thấy chấm đèn).
  3. Xe bị khuất một phần sau đuôi xe khác hoặc sau dải phân cách.
  4. Xe di chuyển với tốc độ cao bị nhòe chuyển động (motion blur).
- **Trường hợp cần người rà lại nhãn tham chiếu trước khi kết luận mô hình sai:**
  Theo `data/DATA.md`, nhãn tham chiếu của tập kiểm thử (`data/test/labels/`) được sinh tự động bởi một mô hình phát hiện đối tượng khác và **chưa được con người rà soát từng box**. Có những trường hợp nhãn tham chiếu sinh ra box sai (ví dụ: gán box `car` lên vệt phản chiếu ánh đèn pha trên mặt đường ướt, gán box lên biển báo phát sáng, hoặc gán box đè lên bóng cây). Khi mô hình YOLOv8n cold start không phát hiện các vùng này, phép đo tự động tính là mô hình bị sai lệch (hoặc false negative nếu nhãn test có box ảo), nhưng trên thực tế mô hình đã dự đoán đúng bản chất vật lý. Do đó, người đánh giá phải trực tiếp mở `outputs/compare_round0.jpg` để kiểm tra mắt xem nhãn tham chiếu có bị ảo hay không trước khi quy kết lỗi cho mô hình.

---

## 3. Chiến lược chọn mẫu

### Giải thích công thức và vai trò các thành phần
Công thức tính điểm ưu tiên chọn mẫu trong `tools/al_select.py`:
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D = 0.5 \cdot U + 0.3 \cdot A + 0.2 \cdot D$$

1. **Độ bất định $U$ (Uncertainty, trọng số $W_U = 0.5$):** 
   Với mỗi box dự đoán có confidence $c \ge 0.05$, độ bất định tính theo $u(c) = 1 - |2c - 1|$. Hàm này đạt giá trị cực đại là $1.0$ khi $c = 0.5$ (mô hình phân vân nhất giữa hai quyết định có phải xe hay không) và tiệm cận $0.0$ khi $c$ gần $0$ hoặc $1$. Giá trị $U$ là trung bình cộng của 5 giá trị $u$ lớn nhất trong frame, đại diện cho mức độ khó và phân vân của những đối tượng thử thách nhất trong ảnh.
2. **Độ mập mờ $A$ (Ambiguity, trọng số $W_A = 0.3$):** 
   Đếm số lượng box có confidence nằm trong khoảng mập mờ $[0.15, 0.50)$, sau đó chuẩn hóa bằng cách chia cho số lượng box mập mờ lớn nhất từng ghi nhận trong toàn bộ pool. Thành phần $A$ ưu tiên những khung hình có mật độ xe phức tạp, nhiều đối tượng gây nhiễu.
3. **Độ đa dạng thời gian $D$ (Diversity, trọng số $W_D = 0.2$):** 
   Tính bằng khoảng cách thời gian từ frame đang xét tới frame đã được gán nhãn gần nhất, chia cho ngưỡng trần $10.0$ giây: $D = \min(\text{gap}, 10.0) / 10.0$. Ở vòng 1 khi chưa có frame nào được gán, $D = 1.0$ cho mọi frame. Ở các vòng tiếp theo, $D$ giúp đẩy thuật toán khám phá các vùng thời gian mới, tránh dồn nhãn vào một đoạn video cục bộ.
4. **Vai trò của `MIN_GAP_S = 2.0` giây:**
   Thuật toán duyệt tham lam (greedy) theo score giảm dần. Nếu một frame ứng viên có khoảng cách thời gian tới bất kỳ frame nào đã được chọn trong cùng một lô nhỏ hơn $2.0$ giây, ứng viên đó sẽ bị loại bỏ. Do camera cố định trên cầu vượt, hai khung hình cách nhau dưới 2 giây có cùng một luồng giao thông và vị trí các xe gần như không đổi. `MIN_GAP_S` ngăn chặn việc gán nhãn các ảnh gần trùng lặp (near-duplicates), giúp tối ưu hóa ngân sách gán nhãn.

### Phân tích các frame dẫn chứng từ `reports/SELECTION.md`
- **3 frame được chọn trong lô 12 ảnh:**
  1. `frame_0182.jpg` (Rank 1, $t=72.8$s, $\text{score}=0.9591$, $U=0.9182$, $A=1.0000$): Có số box mập mờ cao nhất pool (18 box), chứa luồng xe hai chiều phức tạp với nhiều xe ở xa chỉ thấy chấm đèn.
  2. `frame_0369.jpg` (Rank 2, $t=147.6$s, $\text{score}=0.9324$, $U=0.9315$, $A=0.8889$): Mật độ xe cao (43 box) và độ bất định lớn ở các làn xe xa cuối video.
  3. `frame_0331.jpg` (Rank 5, $t=132.4$s, $\text{score}=0.9154$, $U=0.8308$, $A=1.0000$): Có tổng số box cao nhất lô (47 box) với nhiều vệt phản quang trên mặt đường cần người phân định.
- **1 frame cân nhắc khác bị loại bỏ:**
  - `frame_0372.jpg` (Rank 6, $\text{score}=0.9101$, $t=148.8$s): Mặc dù có điểm số rất cao (xếp thứ 6/268, cao hơn cả frame được chọn ở Rank 7 là `frame_0312.jpg`), nhưng vì chỉ cách `frame_0369.jpg` ($t=147.6$s) đúng $1.2$ giây ($< 2.0$s), frame này đã bị loại bỏ theo luật `MIN_GAP_S`. Quyết định này giúp tiết kiệm công rà 42 box cho người dán nhãn mà không làm giảm tính đa dạng của tập huấn luyện.

### Điểm bất định có chứng minh ảnh đó sẽ cải thiện mô hình không?
**Không.** Điểm bất định cao không phải là bằng chứng đảm bảo mô hình sẽ tốt lên sau khi học, vì những lý do sau:
1. **Nhiễu không thể học (Aleatoric uncertainty):** Một vùng ảnh có thể có $u(c) \approx 1.0$ do bị chói lóa đèn pha, nhòe chuyển động quá nặng, hoặc chỉ là bóng cây/vệt nước phản chiếu. Con người gán nhãn các vùng này có thể vô tình đưa thêm nhãn nhiễu vào tập huấn luyện.
2. **Không phát hiện được lỗi "tự tin sai" (Overconfident errors):** Nếu mô hình nhận nhầm một vệt đèn là xe với confidence $0.99$, hoặc bỏ sót hoàn toàn một chiếc xe (không phát hiện và không sinh ra box nào), thì $u(c) \approx 0$ và điểm bất định không thể cảnh báo được các ca lỗi nghiêm trọng này.
3. **Nguy cơ quên thảm họa (Catastrophic forgetting):** Khi chỉ bổ sung một lượng nhỏ ảnh khó (12 ảnh) vào huấn luyện mà không có cơ chế giữ lại phân bố dữ liệu tổng quát, mô hình rất dễ bị overfit vào các trường hợp dị biệt và suy thoái trên toàn bộ tập test.

---

## 4. Các vòng học chủ động (active learning)

### Bảng kết quả so sánh qua các vòng từ `reports/rounds_table.md`

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 260 | 0.335 | -0.436 | 1.000 | 0.052 | 0.099 | 0.000 | 0.044 | 0.195 |

### Mức độ sửa nhãn gợi ý vòng 1 (từ `outputs/round1_diff.md`)
- **Tổng số ảnh:** 12 ảnh pool.
- **Model đề xuất ban đầu:** 169 box.
- **Sau khi sửa thủ công:** 260 box.
- **Phân loại thao tác:**
  - `accepted` (giữ nguyên, IoU $\ge 0.85$): **134 box**
  - `edited` (chỉnh lại vị trí/kích thước, $0.50 \le \text{IoU} < 0.85$): **20 box**
  - `deleted` (xóa box giả - FP của mô hình): **15 box**
  - `added` (vẽ thêm xe bị bỏ sót - FN của mô hình): **106 box**
  - **Tỷ lệ chấp nhận (Accept rate):** **79%**

### Biến động số đo AP50 và các nhóm xe
- **Biến động AP50:** Sau vòng 1, AP50 giảm mạnh từ **0.771** xuống **0.335** ($\Delta = -0.436$). Mặc dù Precision@0.25 đạt tuyệt đối **1.000** (không có bất kỳ phát hiện giả FP nào trên tập test), nhưng Recall@0.25 sụt giảm nghiêm trọng từ **0.489** xuống chỉ còn **0.052** (chỉ phát hiện được 21 box, bỏ sót tới 382 box!).
- **Chi tiết các nhóm xe:**
  - **Small:** Recall rơi từ **0.182** về **0.000** (hoàn toàn không phát hiện được bất kỳ chiếc xe nhỏ nào).
  - **Medium:** Recall rơi từ **0.547** xuống **0.044** (bắt được 13/296 box).
  - **Large:** Recall rơi từ **0.561** xuống **0.195** (bắt được 8/41 box).

### Phân tích ca kết quả đổi sau fine-tune từ `outputs/compare_round1.jpg`
So sánh trực quan giữa `compare_round0.jpg` và `compare_round1.jpg` trên 20 ảnh test cho thấy một sự thay đổi rất rõ rệt:
- Ở vòng 0, mô hình dự đoán khá nhiều box bao phủ từ cự ly gần đến cự ly trung bình (dù một số box có confidence dao động quanh 0.3 - 0.5).
- Ở vòng 1, sau khi fine-tune 50 epochs với 12 ảnh (260 box), mô hình trở nên "siêu thận trọng" (over-conservative). Nó triệt tiêu gần như toàn bộ các box ở cự ly xa và cự ly trung bình, chỉ dự đoán một vài chiếc xe lớn nhất ở làn đường sát mép dưới camera.
- **Lý do có thể kiểm chứng:** Đây là biểu hiện kinh điển của hiện tượng **Catastrophic Forgetting** kết hợp với **Overfitting trên tập dữ liệu nhỏ**. Mô hình YOLOv8n vốn được tiền huấn luyện trên hàng chục ngàn ảnh COCO với đa dạng góc nhìn. Khi fine-tune với learning rate mặc định mà không đóng băng (freeze) các tầng backbone, chỉ 12 ảnh với 260 box đã phá vỡ các đặc trưng tổng quát ban đầu. Mô hình co cụm lại, chỉ dám tự tin ở những đối tượng có tín hiệu mạnh nhất.

### Phân biệt 3 mức thông tin: Quan sát độc lập, Lỗi pre-label đã sửa và Kết quả sau train
1. **Quan sát độc lập (`reports/BLIND_SCAN.md` trên `frame_0099.jpg`):** Người rà nhãn đếm thấy 26 xe bằng mắt thường và ghi nhận hai vị trí AI dễ bỏ sót/vẽ sai: xe ở sát mép phải màn hình (bị cắt nửa thân sau) và xe ở trên cùng bên trái bị cây che.
2. **Lỗi pre-label đã sửa (`outputs/round1_diff.md` & `reports/REVIEW_LOG.csv`):** AI chỉ đề xuất 13 box cho `frame_0099.jpg`. Người gán nhãn đã thêm 7 box (trong đó có chiếc xe sát mép phải và xe bị che khuất), sửa lại 3 box cho ôm khít thân xe, đưa tổng số box lên 20. Trên toàn bộ 12 ảnh, người rà đã loại bỏ 15 box giả (chủ yếu là vệt sáng phản chiếu trên mặt đường) và bổ sung 106 box bị bỏ sót.
3. **Kết quả mô hình sau train (`outputs/metrics_round1.json`):** Mặc dù dữ liệu gán nhãn đã được con người chỉnh sửa rất chuẩn mực theo quy tắc, mô hình sau khi huấn luyện lại bị suy giảm Recall trên tập test do vấn đề tối ưu hóa kích thước lô và siêu tham số huấn luyện. Điều này chứng minh: **Nhãn huấn luyện chất lượng cao là điều kiện cần, nhưng chưa đủ nếu kỹ thuật huấn luyện mô hình chưa phù hợp với quy mô dữ liệu nhỏ.**

### Mô tả một ca khó theo guideline
Ca xe ở cự ly xa chỉ nhìn thấy hai chấm đèn đỏ hoặc trắng, phần thân xe chìm trong bóng tối. Theo quy tắc trong `GUIDELINE_LABEL.md`:
- Nếu thân xe tối nhưng vẫn phán đoán được đường viền quanh cụm đèn, người gán nhãn phải vẽ box ôm trọn phần thân xe đoán được, tuyệt đối không chỉ khoanh riêng hai chấm đèn và không vẽ trùm lên vệt sáng rọi xuống mặt đường.
- Nếu xe ở quá xa sát đường chân trời với chiều cao box dưới 16 pixel, quy tắc cho phép bỏ qua không bắt buộc phải gán, vì các box này sẽ được hàm đánh giá (`tools/det_eval.py`) tự động bỏ qua khi chấm điểm nhằm tránh tranh cãi về độ chủ quan của mắt người.

---

## 5. Kết luận và giới hạn

### Đánh giá kết quả vòng 1 và quyết định dừng / tiếp tục
So với cold start, vòng 1 cho thấy sự đánh đổi cực đoan: Precision tăng từ 0.925 lên 1.000, nhưng Recall giảm từ 0.489 xuống 0.052, kéo tụt AP50 từ 0.771 xuống 0.335.
- **Quyết định:** Trong điều kiện thực tế của buổi lab thực hành, chúng ta dừng lại ở vòng 1 để tập trung phân tích bản chất của hiện tượng suy giảm hiệu năng này. Nếu tiếp tục làm vòng 2, việc chỉ đơn giản gán thêm 12 ảnh theo cách cũ sẽ khó đảo ngược tình thế nếu không can thiệp vào cấu hình huấn luyện.
- **Đề xuất hai ca còn yếu hoặc bất định cho vòng tiếp theo:**
  1. **Ca xe kích thước nhỏ (Small) và xe ở xa:** Điểm đo cho thấy $R_{\text{small}} = 0.000$. Cần ưu tiên các frame có nhiều xe nhỏ ở các làn xa (như `frame_0218.jpg` tại $t=87.2$s hoặc `frame_0114.jpg` tại $t=45.6$s).
     - *Chi phí rà nhãn:* Khá cao vì mỗi ảnh có trên 30 xe, người gán nhãn phải phóng to từng cụm đèn mờ để phân định.
     - *Nguy cơ ảnh gần trùng:* Cần chọn frame cách các frame đã gán tối thiểu $> 3$ giây để đảm bảo tính đa dạng thời gian $D$.
  2. **Ca xe lớn, xe tải, xe buýt có hình khối dị biệt:** Hiện tại $R_{\text{large}}$ chỉ đạt 0.195. Cần bổ sung các frame có xe tải thùng dài hoặc xe buýt (như `frame_0026.jpg` tại $t=10.4$s, 31 boxes).
     - *Chi phí rà nhãn:* Trung bình (khoảng 30–35 box).
     - *Nguy cơ ảnh gần trùng:* Thấp, vì phân bố ở mốc thời gian đầu video ($t=10.4$s), hoàn toàn cách xa các ảnh vòng 1 (vốn tập trung nhiều quanh $t=70$s – $150$s).

### Tác động của các giới hạn thực nghiệm đến kết luận
1. **Tập kiểm thử chỉ có 20 ảnh (403 box tham chiếu):** Đây là kích thước mẫu khá nhỏ trong thị giác máy tính. Sai số của một vài khung hình có thể gây biến động lớn đến chỉ số AP50 tổng thể.
2. **Quy tắc bỏ qua box $< 16$ pixel:** Giúp loại bỏ các tranh cãi chủ quan ở vùng chân trời xa xôi, nhưng đồng thời khiến mô hình không được đánh giá ở cự ly phát hiện sớm (early warning detection).
3. **Nhãn tham chiếu do mô hình AI tạo ra:** Nhãn test chưa qua rà soát thủ công của con người nên vẫn chứa các lỗi cố hữu của mô hình nguồn (bắt nhầm vệt phản quang, bỏ sót xe tối). Do đó, chỉ số AP50 phản ánh mức độ trùng khớp giữa mô hình của chúng ta với mô hình tạo nhãn tham chiếu, chứ không đại diện tuyệt đối cho độ chính xác thực tế ngoài đời thực.

### Các bước kiểm tra cần làm trước khi huấn luyện thêm nếu AP50 giảm
Nếu AP50 bị giảm sau một vòng học chủ động, kỹ sư AI cần thực hiện quy trình kiểm tra 3 bước trước khi tiếp tục train:
1. **Kiểm tra chất lượng nhãn đã sửa (`sanity check labels`):** Chạy script kiểm tra xem có box nào bị sai định dạng tọa độ, sai class id (bắt buộc phải là `0`), hoặc người gán nhãn có vô tình vẽ box trùm lên vệt sáng đèn pha trên mặt đường hay không.
2. **Điều chỉnh chiến lược huấn luyện (Training & Regularization):**
   - Đóng băng các tầng đặc trưng trích xuất (freeze backbone) của YOLOv8n, chỉ cho phép fine-tune các tầng detection head để giữ lại kiến thức tổng quát từ COCO.
   - Giảm tốc độ học (`lr0`) và áp dụng cơ chế học từ từ (learning rate warmup) để tránh làm hỏng trọng số ban đầu.
   - Giảm số lượng epoch (ví dụ từ 50 xuống 15–20 epoch) để ngăn chặn hiện tượng overfitting trên 12 ảnh.
   - Bổ sung kỹ thuật Replay buffer (trộn thêm một phần dữ liệu tổng quát ban đầu vào các batch huấn luyện).
3. **Đánh giá lại ngưỡng tin cậy (Confidence Threshold):** Kiểm tra đường cong Precision-Recall; có thể mô hình vẫn nhận diện được xe nhưng gán mức confidence thấp (ví dụ 0.15 - 0.20), dẫn đến việc bị bộ lọc `conf = 0.25` loại bỏ hoàn toàn trong phép tính Recall.
