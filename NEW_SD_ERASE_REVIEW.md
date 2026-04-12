# Review new_sd_erase.py (Logic-first, no implementation yet)

## Mục tiêu tài liệu
- Tổng hợp các vấn đề trước khi sửa code.
- Ưu tiên logic thuật toán và hành vi runtime trước.
- Chưa bàn chi tiết triển khai, chỉ chốt hướng giải quyết.

## Tóm tắt nhanh
- Ý tưởng Localized + SPEED là hợp lý.
- File hiện tại chưa đạt mức module ổn định cho continual editing.
- Rủi ro chính nằm ở logic cập nhật nhiều concept và tính nhất quán giữa API với hành vi thực tế.

## Vấn đề logic quan trọng

### 1) Multi-concept update đang có nguy cơ ghi đè, không tích lũy đúng
Hiện tượng:
- Vòng lặp concept lấy trọng số gốc để tính delta, sau đó ghi vào model.
- Khi sang concept tiếp theo, một số layer có thể bị chỉnh lại từ nền cũ thay vì nối tiếp từ trạng thái đã chỉnh.

Tác động:
- Kết quả continual hoặc multi-concept trong một lần chạy có thể lệch mong đợi.
- Concept xử lý sau có thể làm yếu hoặc thay đổi hiệu ứng concept trước.

Hướng giải quyết:
- Quy định rõ semantics: cumulative editing theo thứ tự concept.
- Delta của concept sau phải tính và áp trên trọng số hiện tại (đã chứa update trước đó), không phải snapshot ban đầu.
- Bổ sung log số layer được chỉnh ở mỗi concept để dễ kiểm định.

### 2) Preserve concepts xuất hiện trong chữ ký hàm nhưng không tham gia tối ưu
Hiện tượng:
- API nhận preserve_concepts nhưng logic chính không dùng.

Tác động:
- Người dùng hiểu rằng có cơ chế giữ nội dung, nhưng thực tế không có.
- Dễ gây nhầm trong báo cáo và khó tái lập thí nghiệm.

Câu hỏi mở cần chốt trước khi sửa:
- Tại sao preserve_concepts không được dùng?
- Đây là chủ đích của phương pháp SPEED rút gọn, hay là phần còn thiếu của bản thiết kế?
- Nếu giữ API hiện tại, ta có bắt buộc phải implement preserve objective hay đổi tên tham số để tránh hiểu sai?

Hướng giải quyết:
- Chọn một trong hai:
  - Nhánh A: Implement preserve thật sự (thêm mục tiêu hoặc regularization tương ứng).
  - Nhánh B: Loại bỏ preserve_concepts khỏi API và tài liệu, ghi rõ bản này chỉ erase.

### 3) Tham số lamb có mặt nhưng chưa thể hiện vai trò rõ trong công thức
Hiện tượng:
- Hàm có lamb nhưng pipeline hiện tại chưa dùng nhất quán để điều tiết mức chỉnh sửa.

Tác động:
- Hyperparameter gây hiểu nhầm, khó tune.

Hướng giải quyết:
- Nếu lamb là tham số quan trọng, cần đưa vào công thức update rõ ràng.
- Nếu chưa dùng, tạm bỏ khỏi CLI hoặc đánh dấu reserved và không quảng bá.

### 4) Localize theo một timestep cố định và một pass
Hiện tượng:
- Chọn t cố định để chụp activation.

Tác động:
- Top-K circuit có thể thiếu ổn định giữa seed, prompt style, hoặc distribution khác.

Hướng giải quyết:
- Dùng scoring ổn định hơn: trung bình nhiều timestep và/hoặc nhiều latent seed.
- Cho phép cấu hình localization mode (fast vs stable).

### 5) Tiêu chí chọn Top-K chưa có cơ chế kiểm soát độ phủ
Hiện tượng:
- Chỉ lấy Top-K lớn nhất theo delta trung bình.

Tác động:
- Có thể quá hẹp cho concept rộng, hoặc quá rộng cho concept hẹp.

Hướng giải quyết:
- Bổ sung ngưỡng theo percentile hoặc min score.
- Báo cáo phân phối score để người dùng chọn K hợp lý.

## Vấn đề kỹ thuật nên xử lý sau logic

### 6) Save path cần an toàn hơn
- Tránh lỗi khi người dùng đưa tên file không có thư mục cha.
- Đây là hardening kỹ thuật, làm sau khi chốt logic update.

### 7) Load previous weights cần chuẩn hóa hành vi continual
- Cần định nghĩa rõ precedence: previous weights nạp trước, edit concept sau.
- Cần log số layer load thành công và mismatch.

### 8) Validation shape và fail-fast
- Cần check sớm tương thích chiều embedding và chiều weight.
- Nếu lệch shape thì báo lỗi sớm, không để vỡ giữa vòng lặp.

## Đề xuất thứ tự xử lý
1. Chốt semantics của preserve_concepts và lamb.
2. Chốt semantics cumulative update cho multi-concept.
3. Chốt chiến lược localization ổn định (timestep/seed aggregation).
4. Mới harden kỹ thuật: path save, load previous, shape checks, logging.

## Tiêu chí sẵn sàng trước khi chạy module
- Multi-concept có tính tích lũy và kiểm chứng được.
- API không có tham số mồ côi.
- Có log đủ để truy dấu từng concept và từng layer.
- Eval sanity pass với 3 case:
  - single concept
  - multi-concept trong một run
  - continual qua nhiều checkpoint
