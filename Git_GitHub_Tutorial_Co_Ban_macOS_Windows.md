# HƯỚNG DẪN GIT & GITHUB TỪ CƠ BẢN ĐẾN WORKFLOW THỰC TẾ
> **Dành cho người mới bắt đầu hoàn toàn — Thực hành song song macOS & Windows**

---

## MỤC LỤC

1. [Phần 1: Git và GitHub là gì?](#phần-1--git-và-github-là-gì)
2. [Phần 2: Hướng dẫn cài đặt Git](#phần-2--hướng-dẫn-cài-đặt-git)
3. [Phần 3: Cấu hình Git ban đầu (Git Config)](#phần-3--cấu-hình-git-ban-đầu-git-config)
4. [Phần 4: Đăng ký và Thiết lập tài khoản GitHub](#phần-4--đăng-ký-và-thiết-lập-tài-khoản-github)
5. [Phần 5: Xác thực tài khoản GitHub (GitHub Authentication với gh CLI)](#phần-5--xác-thực-tài-khoản-github-github-authentication)
6. [Phần 6: Tạo Repository đầu tiên trên GitHub](#phần-6--tạo-repository-đầu-tiên-trên-github)
7. [Phần 7: Tải Repository về máy (git clone)](#phần-7--tải-repository-về-máy-git-clone)
8. [Phần 8: Hiểu cấu trúc và 3 trạng thái cốt lõi của Git](#phần-8--hiểu-cấu-trúc-và-3-trạng-thái-cốt-lõi-của-git)
9. [Phần 9: Kiểm tra trạng thái dự án (git status)](#phần-9--kiểm-tra-trạng-thái-dự-án-git-status)
10. [Phần 10: Đưa thay đổi vào vùng chờ (git add)](#phần-10--đưa-thay-đổi-vào-vùng-chờ-git-add)
11. [Phần 11: Đóng gói và lưu vết lịch sử (git commit)](#phần-11--đóng-gói-và-lưu-vết-lịch-sử-git-commit)
12. [Phần 12: Đẩy dữ liệu lên máy chủ GitHub (git push)](#phần-12--đẩy-dữ-liệu-lên-máy-chủ-github-git-push)
13. [Phần 13: Lấy dữ liệu mới nhất từ GitHub về (git pull)](#phần-13--lấy-dữ-liệu-mới-nhất-từ-github-về-git-pull)
14. [Phần 14: Phân biệt rõ rệt git clone vs git pull](#phần-14--phân-biệt-rõ-rệt-git-clone-vs-git-pull)
15. [Phần 15: Quản lý liên kết máy chủ từ xa (git remote)](#phần-15--quản-lý-liên-kết-máy-chủ-từ-xa-git-remote)
16. [Phần 16: Làm việc với nhánh cơ bản (Branch & Merge)](#phần-16--làm-việc-với-nhánh-cơ-bản-branch--merge)
17. [Phần 17: So sánh chi tiết thay đổi (git diff)](#phần-17--so-sánh-chi-tiết-thay-đổi-git-diff)
18. [Phần 18: Loại trừ file rác với .gitignore](#phần-18--loại-trừ-file-rác-với-gitignore)
19. [Phần 19: Thực hành Workflow thực tế hoàn chỉnh từ A đến Z](#phần-19--thực-hành-workflow-thực-tế-hoàn-chỉnh-từ-a-đến-z)
20. [Phần 20: Bảng tra cứu các lệnh Git cốt lõi (Cheat Sheet)](#phần-20--bảng-tra-cứu-các-lệnh-git-cốt-lõi-cheat-sheet)
21. [Phần 21: Xử lý sự cố & Lỗi thường gặp (Troubleshooting)](#phần-21--xử-lý-sự-cố--lỗi-thường-gặp-troubleshooting)

---

## BẢNG QUY ƯỚC MÔI TRƯỜNG THỰC HÀNH

Tài liệu này được biên soạn để bạn có thể vừa kiểm tra trên **macOS**, vừa hướng dẫn người học trên **Windows**. Các câu lệnh được hiển thị song song:

| Hệ điều hành | Ứng dụng dòng lệnh | Giao diện Shell |
| :--- | :--- | :--- |
| **macOS** | **Terminal** | Zsh hoặc Bash |
| **Windows** | **PowerShell** hoặc **Windows Terminal** | PowerShell 7 / Windows PowerShell |

> [!NOTE]
> **Quy ước:**
> * Khi câu lệnh hoàn toàn giống nhau giữa hai hệ điều hành, tài liệu sẽ ghi chú rõ: *(Command này giống nhau trên macOS và Windows)*.
> * Các giá trị nằm trong dấu ngoặc nhọn `<...>` như `<username>`, `<repository-url>`, `<tên-file>` là **placeholder (thông tin thay thế)**. Bạn hãy thay bằng thông tin thực tế của mình mà không giữ lại dấu ngoặc nhọn `< >`.

---

# PHẦN 1 — GIT VÀ GITHUB LÀ GÌ?

## 1. Git là gì?
* **Git** là một **Hệ thống Quản lý Phiên bản Phân tán** (*Distributed Version Control System - DVCS*).
* Hiểu đơn giản: Git giống như một "cỗ máy thời gian" cho mã nguồn. Mỗi khi bạn lưu lại một mốc phát triển, Git sẽ chụp lại toàn bộ trạng thái của thư mục dự án. Nếu bạn lỡ tay xóa nhầm file hoặc code bị lỗi nghiêm trọng, bạn luôn có thể quay ngược về phiên bản hoạt động tốt trước đó.
* **Git chạy hoàn toàn cục bộ (*offline*) trên máy tính của bạn.** Bạn không cần internet để sử dụng Git.

## 2. GitHub là gì?
* **GitHub** là một **nền tảng dịch vụ đám mây (*cloud service*)** trực tuyến, được thiết kế chuyên biệt để lưu trữ và chia sẻ các kho mã nguồn được quản lý bằng Git.
* GitHub giúp bạn sao lưu mã nguồn an toàn trên mạng, cộng tác làm việc cùng đồng nghiệp, xem lại lịch sử thay đổi thông qua giao diện web trực quan.

## 3. Phân biệt Git và GitHub

```text
┌─────────────────────────────────────────────────────────────┐
│                    BẢN CHẤT KHÁC BIỆT                       │
├──────────────────────────────┬──────────────────────────────┤
│             GIT              │            GITHUB            │
├──────────────────────────────┼──────────────────────────────┤
│ Là phần mềm công cụ          │ Là dịch vụ website trực tuyến│
│ Cài đặt và chạy trên máy tính│ Chạy trên máy chủ đám mây    │
│ Hoạt động không cần Internet │ Cần Internet để truy cập     │
│ Lưu trữ phiên bản cục bộ     │ Chia sẻ và sao lưu mã nguồn  │
└──────────────────────────────┴──────────────────────────────┘
```

## 4. Local Repository và Remote Repository
* **Local Repository (Kho lưu trữ cục bộ):** Nằm trực tiếp trên ổ cứng máy tính cá nhân của bạn (bên trong thư mục ẩn có tên `.git`). Chỉ có bạn truy cập được.
* **Remote Repository (Kho lưu trữ từ xa):** Nằm trên máy chủ của GitHub trên internet. Nhiều người có thể cùng kết nối tới đây để đồng bộ mã nguồn.

```text
  [Máy tính cá nhân của bạn]                 [Đám mây GitHub]
┌─────────────────────────────┐           ┌─────────────────────────────┐
│      Local Repository       │ ──push──> │      Remote Repository      │
│   (Thư mục dự án + .git)    │ <──pull── │   (github.com/<user>/repo)  │
└─────────────────────────────┘           └─────────────────────────────┘
```

---

# PHẦN 2 — HƯỚNG DẪN CÀI ĐẶT GIT

Trước khi làm việc, máy tính cần được cài đặt phần mềm Git.

## 1. Kiểm tra xem Git đã có sẵn chưa

**macOS — Terminal**
```bash
git --version
```

**Windows — PowerShell**
```powershell
git --version
```

*(Command này giống nhau trên macOS và Windows)*

* **Kết quả kỳ vọng:** Nếu đã có Git, màn hình sẽ hiển thị phiên bản, ví dụ: `git version 2.43.0`.
* Nếu hiển thị thông báo lỗi lệnh không tồn tại (*command not found* hoặc *is not recognized*), hãy tiến hành cài đặt theo các bước bên dưới.

---

## 2. Cài đặt trên macOS

Trên macOS, cách chuẩn nhất và đơn giản nhất là cài đặt công cụ **Xcode Command Line Tools** (đã bao gồm sẵn Git do Apple tối ưu):

```bash
xcode-select --install
```

Một cửa sổ thông báo sẽ xuất hiện trên màn hình, bạn nhấn nút **Install** và chờ hệ thống tải về hoàn tất.

*Cách thay thế (nếu bạn sử dụng Homebrew):*
```bash
brew install git
```

Sau khi cài xong, kiểm tra lại bằng lệnh:
```bash
git --version
```

---

## 3. Cài đặt trên Windows

### Cách 1: Sử dụng công cụ Winget (Khuyên dùng — Nhanh nhất)
Mở cửa sổ **PowerShell** và gõ lệnh:

```powershell
winget install --id Git.Git -e --source winget
```

### Cách 2: Cài đặt thủ công bằng bộ cài chính thức
1. Mở trình duyệt và truy cập trang chủ tải Git: [https://git-scm.com/download/win](https://git-scm.com/download/win).
2. Tải về file bộ cài đặt (thường là bản *64-bit Git for Windows Setup* dạng file `.exe`).
3. Mở file vừa tải và tiến hành cài đặt. Bạn có thể giữ toàn bộ tùy chọn mặc định bằng cách bấm **Next** liên tục cho tới khi hoàn thành (**Finish**).

> [!IMPORTANT]
> **Bước bắt buộc trên Windows:** Sau khi cài đặt hoàn tất, bạn **PHẢI TẮT HẲN** cửa sổ PowerShell đang mở và khởi động lại một cửa sổ PowerShell mới để hệ thống cập nhật biến môi trường nhận diện lệnh `git`.

Kiểm tra lại trên PowerShell mới:
```powershell
git --version
```

---

# PHẦN 3 — CẤU HÌNH GIT BAN ĐẦU (GIT CONFIG)

> [!IMPORTANT]
> **Khái niệm then chốt:** Lệnh `git config` dùng để khai báo thông tin tác giả (Họ tên và Email) sẽ được gắn vào mỗi mốc lịch sử (commit) mà bạn tạo ra.
> **Lệnh này KHÔNG PHẢI là đăng nhập vào tài khoản GitHub.**

## 1. Cấu hình Tên và Email tác giả

Hãy thay thế `"Your Name"` bằng tên của bạn và `"your@email.com"` bằng địa chỉ email chính xác mà bạn dùng để đăng ký tài khoản GitHub.

**macOS — Terminal**
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

**Windows — PowerShell**
```powershell
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

*(Command này giống nhau trên macOS và Windows)*

## 2. Đặt tên nhánh mặc định là `main`
Theo tiêu chuẩn hiện đại của Git và GitHub, nhánh chính thống nhất được đặt tên là `main` (thay vì `master` như trước đây):

**macOS — Terminal**
```bash
git config --global init.defaultBranch main
```

**Windows — PowerShell**
```powershell
git config --global init.defaultBranch main
```

*(Command này giống nhau trên macOS và Windows)*

## 3. Kiểm tra lại toàn bộ cấu hình đã lưu

**macOS — Terminal**
```bash
git config --global --list
```

**Windows — PowerShell**
```powershell
git config --global --list
```

*(Command này giống nhau trên macOS và Windows)*

**Kết quả hiển thị ví dụ:**
```text
user.name=Nguyen Van A
user.email=nguyenvana@gmail.com
init.defaultbranch=main
```

---

# PHẦN 4 — ĐĂNG KÝ VÀ THIẾT LẬP TÀI KHOẢN GITHUB

Để chia sẻ mã nguồn và làm việc nhóm, bạn cần có tài khoản trên trang web GitHub.

## Các bước thực hiện:
1. Mở trình duyệt web và truy cập: [https://github.com/](https://github.com/)
2. Nhấn nút **Sign up** ở góc trên bên phải màn hình.
3. Điền các thông tin:
   * **Email:** Nhập email của bạn (trùng với email đã cấu hình ở Phần 3).
   * **Password:** Tạo mật khẩu mạnh.
   * **Username:** Đặt tên tài khoản (viết liền, không dấu, ví dụ: `nguyenvana-dev`).
4. Hoàn thành câu đố bảo mật (captcha) và nhấn **Create account**.
5. Mở hộp thư điện tử của bạn, tìm thư xác nhận từ GitHub và nhập mã xác minh (Launch code).
6. Giữ tài khoản luôn đăng nhập trên trình duyệt web của bạn.

---

# PHẦN 5 — XÁC THỰC TÀI KHOẢN GITHUB (GITHUB AUTHENTICATION)

Trước đây, Git cho phép người dùng nhập trực tiếp mật khẩu tài khoản GitHub khi đẩy mã nguồn. Tuy nhiên, từ tháng 8/2021, GitHub đã chấm dứt hỗ trợ mật khẩu trực tiếp vì lý do bảo mật. 

Cách đơn giản, an toàn và trực quan nhất hiện nay cho người mới bắt đầu là sử dụng **GitHub CLI (`gh`)**.

> [!NOTE]
> * `git config`: Đăng ký tên tác giả xuất hiện trong lịch sử code.
> * `gh auth login`: Đăng nhập cấp quyền cho máy tính kết nối với máy chủ GitHub.

## 1. Kiểm tra công cụ GitHub CLI

**macOS — Terminal**
```bash
gh --version
```

**Windows — PowerShell**
```powershell
gh --version
```

*(Command này giống nhau trên macOS và Windows)*

Nếu chưa có công cụ này, hãy cài đặt như sau:

### Cài đặt trên macOS:
```bash
brew install gh
```

### Cài đặt trên Windows (PowerShell):
```powershell
winget install --id GitHub.cli
```
*(Lưu ý: Nếu vừa cài trên Windows, hãy tắt và mở lại PowerShell).*

---

## 2. Tiến hành đăng nhập xác thực bằng trình duyệt

Gõ lệnh sau vào cửa sổ dòng lệnh:

**macOS — Terminal**
```bash
gh auth login
```

**Windows — PowerShell**
```powershell
gh auth login
```

*(Command này giống nhau trên macOS và Windows)*

### Các bước lựa chọn trên màn hình:
Dùng phím mũi tên lên/xuống và phím **Enter** để chọn đúng các mục sau:

1. `What account do you want to log into?`
   * Chọn: **GitHub.com** (nhấn Enter).
2. `What is your preferred protocol for Git operations on this host?`
   * Chọn: **HTTPS** (nhấn Enter).
3. `Authenticate Git with your GitHub credentials?`
   * Chọn: **Yes** (nhấn Enter).
4. `How would you like to authenticate GitHub CLI?`
   * Chọn: **Login with a web browser** (nhấn Enter).
5. Màn hình sẽ cung cấp cho bạn một mã một lần (one-time code) gồm 8 ký tự, ví dụ: `A1B2-C3D4`.
   * Hãy bôi đen và copy mã này.
   * Nhấn phím **Enter** trên bàn phím. Cửa sổ trình duyệt web mặc định sẽ tự động mở ra trang xác thực của GitHub.
   * Dán mã 8 ký tự vào ô trống trên web -> Nhấn **Continue** -> Nhấn **Authorize github**.

---

## 3. Kiểm tra trạng thái đăng nhập

**macOS — Terminal**
```bash
gh auth status
```

**Windows — PowerShell**
```powershell
gh auth status
```

*(Command này giống nhau trên macOS và Windows)*

**Kết quả thành công:**
```text
✓ Logged in to github.com account <username> (https)
- Active account: true
- Git operations protocol: https
```

---

## 4. Mở rộng: Phân biệt chi tiết giữa HTTPS và SSH Key (Nên chọn loại nào?)

Khi làm việc với GitHub, bạn sẽ thường xuyên bắt gặp hai tùy chọn giao thức kết nối: **HTTPS** và **SSH**. Cả hai đều phục vụ chung một mục đích là truyền nhận mã nguồn an toàn giữa máy tính của bạn và máy chủ GitHub, nhưng cơ chế hoạt động và trải nghiệm thiết lập hoàn toàn khác nhau.

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                      HAI CON ĐƯỜNG KẾT NỐI GITHUB                       │
├────────────────────────────────────┬────────────────────────────────────┤
│               HTTPS                │                SSH                 │
├────────────────────────────────────┼────────────────────────────────────┤
│ https://github.com/user/repo.git   │ git@github.com:user/repo.git       │
│ Kết nối qua giao thức Web (Port 443│ Kết nối qua giao thức mã hóa SSH   │
│ Xác thực qua tài khoản / Token/ CLI│ Xác thực qua cặp khóa (Public &    │
│                                    │ Private Key)                       │
│ Rất dễ cài đặt cho người mới       │ Cần biết tạo và quản lý SSH Key    │
│ Không bao giờ bị chặn bởi tường lửa│ Có thể bị chặn ở mạng công ty kín  │
└────────────────────────────────────┴────────────────────────────────────┘
```

### A. Giao thức HTTPS (Khuyên dùng cho người mới bắt đầu)

* **Định dạng đường dẫn:** Bắt đầu bằng `https://`, ví dụ:
  `https://github.com/<username>/<repo-name>.git`
* **Cách thức hoạt động:**
  * Hoạt động qua cổng mạng tiêu chuẩn 443 (cổng truy cập website bảo mật), hầu như không bao giờ bị tường lửa ở trường học, công ty hay quán cafe chặn.
  * Trước đây, khi dùng HTTPS bạn phải gõ tài khoản/mật khẩu hoặc tạo Personal Access Token (PAT) thủ công rất rườm rà. Nhưng hiện nay, khi kết hợp với **GitHub CLI (`gh auth login`)**, bạn chỉ cần đăng nhập qua trình duyệt web một lần duy nhất. Trình quản lý chứng chỉ sẽ tự động lưu lại phiên đăng nhập, các lần push/pull tiếp theo diễn ra hoàn toàn tự động.
* **Ưu điểm:**
  * Cực kỳ dễ thiết lập (chỉ mất 1–2 phút thao tác trên trình duyệt).
  * Thân thiện với người mới bắt đầu, không lo gõ sai lệnh quản lý file khóa.
  * Hoạt động trơn tru trên mọi mạng internet.
* **Nhược điểm:**
  * Token xác thực có thể hết hạn sau một thời gian dài không sử dụng (khi đó chỉ cần chạy lại `gh auth login`).

---

### B. Giao thức SSH (Secure Shell)

* **Định dạng đường dẫn:** Bắt đầu bằng `git@github.com:`, ví dụ:
  `git@github.com:<username>/<repo-name>.git`
* **Cách thức hoạt động:**
  * SSH hoạt động dựa trên cơ chế **Mã hóa bất đối xứng (Asymmetric Cryptography)** với một **cặp chìa khóa (Key Pair)**:
    1. **Private Key (Khóa riêng tư / bí mật):** Nằm an toàn bên trong máy tính của bạn (thường tại `~/.ssh/id_ed25519`). **Tuyệt đối không gửi file này cho bất kỳ ai!**
    2. **Public Key (Khóa công khai):** Là file có đuôi `.pub` (ví dụ `~/.ssh/id_ed25519.pub`). Bạn dán nội dung file này lên trang web GitHub (`Settings` -> `SSH and GPG keys`).
  * Mỗi khi bạn push hoặc pull, GitHub và máy tính của bạn sẽ tự "bắt tay" xác thực danh tính thông qua cặp khóa này mà không cần truyền bất kỳ mật khẩu nào qua mạng.
* **Ưu điểm:**
  * Bảo mật cực cao, không lo bị lộ mật khẩu.
  * Khóa SSH tồn tại vĩnh viễn trên máy cho đến khi bạn chủ động xóa đi.
  * Tiêu chuẩn bắt buộc khi làm việc với máy chủ Linux từ xa (VPS, Cloud Server, CI/CD tự động).
* **Nhược điểm:**
  * Khá phức tạp cho người mới: Phải biết dùng dòng lệnh sinh key, tìm đúng đường dẫn file và dán lên web.
  * Một số mạng nội bộ công ty hoặc trường học bảo mật gắt gao có thể chặn cổng 22 (cổng mặc định của SSH).

---

### C. Bảng so sánh chi tiết và Lời khuyên lựa chọn

| Tiêu chí | HTTPS (với GitHub CLI) | SSH Key |
| :--- | :--- | :--- |
| **Định dạng URL** | `https://github.com/<user>/<repo>.git` | `git@github.com:<user>/<repo>.git` |
| **Độ khó thiết lập** | ⭐ Rất dễ (Đăng nhập qua web browser) | ⭐⭐⭐ Trung bình (Tạo key qua terminal) |
| **Cổng kết nối mạng** | Cổng 443 (Hiếm khi bị chặn) | Cổng 22 (Đôi khi bị chặn ở mạng kín) |
| **Thời hạn sử dụng** | Có thể cần cấp lại phiên đăng nhập | Dùng vĩnh viễn đến khi xóa key |
| **Đối tượng phù hợp** | **Người mới bắt đầu**, học sinh, sinh viên | **Lập trình viên kinh nghiệm**, quản trị server DevOps |

> [!TIP]
> **Lời khuyên thực tế:**
> * Nếu bạn là **người mới bắt đầu**: Hãy kiên định chọn **HTTPS** và đăng nhập bằng **`gh auth login`** như đã hướng dẫn. Đây là giải pháp nhanh chóng nhất, ít lỗi phát sinh nhất để bạn tập trung trọn vẹn vào việc học Git workflow.
> * Bạn chỉ nên chuyển sang học cấu hình **SSH Key** khi bắt đầu làm việc với máy chủ từ xa (VPS) hoặc khi công ty yêu cầu chuẩn SSH.

---

### D. (Tham khảo thêm) Cách tạo và cấu hình SSH Key nếu muốn sử dụng

Dưới đây là các bước nhanh nếu bạn muốn tự tay cấu hình SSH Key:

#### Bước 1: Tạo cặp SSH Key trên máy tính

**macOS — Terminal**
```bash
ssh-keygen -t ed25519 -C "your@email.com"
```

**Windows — PowerShell**
```powershell
ssh-keygen -t ed25519 -C "your@email.com"
```

*(Command này giống nhau trên macOS và Windows)*

* Khi được hỏi nơi lưu file (`Enter file in which to save the key`), nhấn **Enter** để đồng ý dùng đường dẫn mặc định.
* Khi được hỏi mật khẩu bảo vệ (`Enter passphrase`), nhấn **Enter** 2 lần để bỏ qua nếu muốn tiện lợi, hoặc nhập mật khẩu nếu cần an toàn tuyệt đối.

#### Bước 2: Đọc và sao chép Public Key

**macOS — Terminal**
```bash
cat ~/.ssh/id_ed25519.pub
```

**Windows — PowerShell**
```powershell
Get-Content ~/.ssh/id_ed25519.pub
```

*Màn hình sẽ hiển thị một dòng bắt đầu bằng `ssh-ed25519 AAAAC3NzaC1...`. Hãy sao chép toàn bộ dòng đó.*

#### Bước 3: Dán Public Key lên GitHub
1. Mở GitHub trên trình duyệt -> Bấm vào **Ảnh đại diện** ở góc trên cùng bên phải -> Chọn **Settings**.
2. Ở menu bên trái, tìm và chọn mục **SSH and GPG keys**.
3. Bấm nút màu xanh lá cây **New SSH key**.
4. Ô **Title**: Đặt tên nhận diện máy tính của bạn (ví dụ: `Laptop-Cua-Toi`).
5. Ô **Key**: Dán toàn bộ chuỗi ký tự đã copy ở Bước 2 vào đây.
6. Bấm nút **Add SSH key**.

#### Bước 4: Kiểm tra kết nối SSH thành công

**macOS & Windows**
```bash
ssh -T git@github.com
```
*Nếu nhận được thông báo: `Hi <username>! You've successfully authenticated...` tức là bạn đã kết nối thành công qua SSH!*

---

# PHẦN 6 — TẠO REPOSITORY ĐẦU TIÊN TRÊN GITHUB

Repository (viết tắt là **Repo**) là một "kho lưu trữ dự án". Mọi file code, hình ảnh, tài liệu và lịch sử phát triển của dự án đều nằm trong Repo này.

```text
Trang chủ GitHub
       ↓
Nút "New" (Repository)
       ↓
Nhập tên Repository
       ↓
Chọn Public hoặc Private
       ↓
Tích chọn "Add a README file"
       ↓
Nhấn "Create repository"
```

## Các bước thao tác trên website GitHub:
1. Đăng nhập vào [https://github.com/](https://github.com/).
2. Nhìn sang góc trên cùng bên phải, bấm vào biểu tượng dấu cộng `+` -> Chọn **New repository** (hoặc bấm nút màu xanh lá cây **New** ở cột bên trái).
3. Thiết lập thông tin kho lưu trữ:
   * **Repository name:** Nhập tên dự án, viết liền không dấu, dùng dấu gạch ngang (ví dụ: `du-an-dau-tien` hoặc `demo-git`).
   * **Description:** Mô tả ngắn gọn về dự án (có thể để trống).
   * **Chế độ hiển thị:**
     * **Public:** Mọi người trên Internet đều có thể xem mã nguồn của bạn. Thích hợp cho dự án học tập, mã nguồn mở, kho portfolio xin việc.
     * **Private:** Chỉ duy nhất bạn (và những người được bạn chỉ định) mới có quyền xem mã nguồn.
   * **Initialize this repository with:** Tích chọn vào ô **Add a README file** *(File README.md là trang bìa giới thiệu tóm tắt dự án của bạn)*.
4. Bấm nút **Create repository** ở dưới cùng.

## Lấy đường dẫn (URL) của Repository vừa tạo
Sau khi tạo xong, bạn đang đứng tại trang chủ của repository:
1. Bấm vào nút màu xanh lá cây có chữ **Code**.
2. Tại đây bạn sẽ thấy các tab lựa chọn giao thức (như đã phân biệt ở Phần 5):
   * **Tab HTTPS (Khuyên dùng cho người mới):** Chọn tab **HTTPS** -> Copy đường link có dạng:
     `https://github.com/<username>/<repository-name>.git`
   * **Tab SSH (Nếu đã cấu hình SSH Key):** Chọn tab **SSH** -> Copy đường link có dạng:
     `git@github.com:<username>/<repository-name>.git`
3. Bấm vào biểu tượng **hai hình vuông lồng nhau** để sao chép đường link bạn đã chọn.

---

# PHẦN 7 — TẢI REPOSITORY VỀ MÁY (GIT CLONE)

Lệnh `git clone` dùng để tải một bản sao đầy đủ của toàn bộ dự án từ trên GitHub về ổ cứng máy tính của bạn.

## 1. Mở thư mục Desktop trên dòng lệnh

Chúng ta sẽ tải dự án về màn hình Desktop để dễ quan sát:

**macOS — Terminal**
```bash
cd ~/Desktop
```

**Windows — PowerShell**
```powershell
cd ~/Desktop
```

> [!NOTE]
> Ký tự ngã `~` đại diện cho thư mục người dùng cá nhân (*Home Directory*). Trên cả macOS Terminal và Windows PowerShell hiện đại, `cd ~/Desktop` đều di chuyển bạn tới màn hình chính máy tính.

---

## 2. Thực hiện tải Repository về máy

Dán đường dẫn HTTPS bạn vừa copy ở Phần 6 vào lệnh sau:

**macOS — Terminal**
```bash
git clone <repository-url>
```

**Windows — PowerShell**
```powershell
git clone <repository-url>
```

*(Command này giống nhau trên macOS và Windows)*

**Ví dụ thực tế:**
```bash
git clone https://github.com/nguyenvana/du-an-dau-tien.git
```

Git sẽ tạo một thư mục mới có tên trùng với tên repository ngay trên Desktop của bạn.

---

## 3. Di chuyển vào bên trong thư mục dự án

**macOS — Terminal**
```bash
cd <repository-name>
```

**Windows — PowerShell**
```powershell
cd <repository-name>
```

*(Command này giống nhau trên macOS và Windows)*

**Ví dụ:**
```bash
cd du-an-dau-tien
```

---

## 4. Mở rộng: Phân biệt `git clone` và `git init` (Hai cách bắt đầu dự án)

Khi bắt đầu một dự án Git, bạn sẽ có hai hướng tiếp cận:

### Cách 1: Sử dụng `git clone` (Workflow chính khuyên dùng cho người mới)
* Bạn tạo Repository rỗng trên GitHub trước, sau đó dùng `git clone <url>` để tải về máy.
* **Ưu điểm vượt trội:** Git tự động thiết lập mọi liên kết với GitHub (`origin`, nhánh `main`), bạn không cần cấu hình thêm bất kỳ liên kết thủ công nào.

### Cách 2: Sử dụng `git init` (Khởi tạo dự án từ máy tính cục bộ)
* Bạn đã có sẵn một thư mục chứa file code trên máy tính từ trước và muốn Git bắt đầu quản lý thư mục đó.
* Di chuyển vào thư mục code và khởi tạo:

**macOS — Terminal**
```bash
git init
```

**Windows — PowerShell**
```powershell
git init
```

*(Command này giống nhau trên macOS và Windows)*

* **Giải thích:** Lệnh `git init` sẽ tạo ra một thư mục ẩn tên là `.git` ngay trong thư mục dự án của bạn. Thư mục này đóng vai trò là Local Repository để lưu trữ lịch sử commit.
* **Lưu ý:** Vì khởi tạo ở máy cục bộ nên dự án này chưa hề kết nối với GitHub. Nếu muốn đưa lên GitHub, sau đó bạn sẽ cần tạo một Repo trên web và chạy thêm lệnh kết nối: `git remote add origin <url>`. Do đó, đối với người mới bắt đầu, **Cách 1 (`git clone`) luôn là phương pháp tiện lợi và ít gặp lỗi nhất!**

---

# PHẦN 8 — HIỂU CẤU TRÚC VÀ 3 TRẠNG THÁI CỐT LÕI CỦA GIT

Đây là phần lý thuyết **quan trọng nhất** để hiểu bản chất vận hành của Git. Mọi thao tác lưu code hàng ngày đều xoay quanh mô hình này.

```text
┌─────────────────────────┐
│    WORKING DIRECTORY    │  ← Nơi bạn trực tiếp gõ code, tạo file, sửa xóa file.
└─────────────────────────┘
             │
             │  git add
             ▼
┌─────────────────────────┐
│      STAGING AREA       │  ← "Chiếc giỏ hàng" chọn lọc các file sẵn sàng đóng gói.
└─────────────────────────┘
             │
             │  git commit
             ▼
┌─────────────────────────┐
│    LOCAL REPOSITORY     │  ← "Kho chứa an toàn" lưu trữ lịch sử commit trên máy bạn.
└─────────────────────────┘
             │
             │  git push
             ▼
┌─────────────────────────┐
│    GITHUB REPOSITORY    │  ← Máy chủ đám mây sao lưu và chia sẻ cho cả nhóm.
└─────────────────────────┘
```

### 1. Working Directory (Thư mục làm việc thực tế)
* Là thư mục vật lý trên ổ cứng của bạn (nơi bạn mở VS Code, gõ code, sửa nội dung file).
* Khi bạn mới tạo hoặc chỉnh sửa file, Git sẽ nhận biết file này đang bị thay đổi nhưng chưa được bảo vệ.

### 2. Staging Area (Vùng chuẩn bị / Index)
* Hãy hình dung bạn đi siêu thị: Bạn nhặt một số món hàng bỏ vào **chiếc giỏ hàng** trước khi ra quầy tính tiền.
* Staging Area chính là chiếc giỏ hàng đó. Bạn dùng lệnh `git add` để chọn lọc những file đã sửa ưng ý đưa vào vùng này, chuẩn bị cho việc lưu lại.

### 3. Local Repository (Kho lưu trữ trên máy)
* Khi bạn quyết định "thanh toán đơn hàng", bạn dùng lệnh `git commit`.
* Toàn bộ những gì nằm trong Staging Area sẽ được đóng gói lại thành một mốc lịch sử vĩnh viễn (commit) và cất vào cơ sở dữ liệu nội bộ của Git (`.git`).

### 4. GitHub Remote Repository (Kho lưu trữ trên đám mây)
* Là nơi bạn dùng `git push` để tải gói hàng commit từ máy tính của mình lên máy chủ GitHub để dự phòng và chia sẻ.

---

# PHẦN 9 — KIỂM TRA TRẠNG THÁI DỰ ÁN (GIT STATUS)

Lệnh `git status` giống như chiếc la bàn trong Git. Hãy tạo thói quen gõ lệnh này thường xuyên.

```bash
git status
```
*(Command này giống nhau trên macOS và Windows)*

## Mục đích của lệnh:
* Cho biết bạn đang đứng ở nhánh (*branch*) nào.
* File nào mới được tạo thêm mà Git chưa theo dõi (*Untracked files*).
* File nào đã bị sửa đổi nội dung (*Changes not staged for commit*).
* File nào đã được đưa vào vùng chuẩn bị (*Changes to be committed*).

## Minh họa kết quả thực tế:

### Trường hợp 1: Dự án sạch sẽ, chưa có chỉnh sửa nào
```text
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
```

### Trường hợp 2: Có file mới tạo nhưng chưa đưa vào Staging Area
*(File `hello.txt` vừa được tạo ra)*
```text
On branch main
Untracked files:
  (use "git add <file>..." to include in what will be committed)
	hello.txt

nothing added to commit but untracked files present (use "git add" to track)
```

---

# PHẦN 10 — ĐƯA THAY ĐỔI VÀO VÙNG CHỜ (GIT ADD)

Sau khi tạo file mới hoặc chỉnh sửa mã nguồn, bạn cần đưa thay đổi đó vào **Staging Area**.

## 1. Cách 1: Thêm một file cụ thể
Sử dụng khi bạn chỉ muốn đóng gói một file duy nhất:

**macOS & Windows**
```bash
git add <tên-file>
```

**Ví dụ:**
```bash
git add hello.txt
```

---

## 2. Cách 2: Thêm toàn bộ các file đã thay đổi (Phổ biến nhất)
Dấu chấm `.` đại diện cho toàn bộ thư mục hiện tại:

**macOS & Windows**
```bash
git add .
```

---

## 3. Kiểm tra lại ngay sau khi add
Gõ lệnh:
```bash
git status
```

**Kết quả hiển thị lúc này:**
```text
On branch main
Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
	new file:   hello.txt
```
*Tên file chuyển sang màu xanh lá cây, thông báo file đã nằm sẵn trong Staging Area, sẵn sàng để đóng gói.*

---

# PHẦN 11 — ĐÓNG GÓI VÀ LƯU VẾT LỊCH SỬ (GIT COMMIT)

Lệnh `git commit` sẽ chụp lại một mốc lịch sử vĩnh viễn trên máy của bạn.

## 1. Thực hiện tạo commit
Cờ `-m` viết tắt của chữ **Message (Thông điệp)**. Bạn luôn phải viết một câu mô tả ngắn gọn về những gì bạn vừa làm:

**macOS & Windows**
```bash
git commit -m "Mô tả nội dung thay đổi"
```

**Ví dụ thực tế:**
```bash
git commit -m "Tao file hello.txt va viet loi chao"
```

**Kết quả hiển thị:**
```text
[main 7f2e1a4] Tao file hello.txt va viet loi chao
 1 file changed, 1 insertion(+)
 create mode 100644 hello.txt
```

> [!TIP]
> **Quy tắc viết Commit Message chuyên nghiệp:**
> * Viết ngắn gọn, rõ nghĩa (dưới 50–72 ký tự).
> * Nêu rõ hành động: *"Add login feature"*, *"Fix header responsive"*, *"Update documentation"*.
> * Tránh viết vô nghĩa như: *"fix"*, *"update"*, *"a"*, *"done"*.

---

## 2. Xem lại lịch sử các commit đã tạo
Lệnh xem toàn bộ lịch sử:

**macOS & Windows**
```bash
git log --oneline
```

**Kết quả hiển thị ví dụ:**
```text
7f2e1a4 (HEAD -> main) Tao file hello.txt va viet loi chao
3b8c9d2 Initial commit
```
*Mỗi commit sẽ có một mã nhận dạng duy nhất gồm 7 ký tự (mã Hash) và thông điệp bạn đã viết.*

---

# PHẦN 12 — ĐẨY DỮ LIỆU LÊN MÁY CHỦ GITHUB (GIT PUSH)

Sau khi tạo commit, dữ liệu mới chỉ được lưu an toàn trong máy tính của bạn (Local Repository). Để đưa lên đám mây GitHub, bạn phải dùng lệnh `git push`.

```text
Local Repository (Máy bạn) ────[ git push ]────> GitHub Repository (Đám mây)
```

## 1. Câu lệnh thực hiện

### Lần đẩy đầu tiên của một nhánh (Khuyên dùng):
Lệnh này giúp thiết lập mối liên kết theo dõi (*upstream tracking*) giữa nhánh trên máy tính và nhánh trên GitHub:

**macOS & Windows**
```bash
git push -u origin main
```

### Các lần đẩy tiếp theo sau này:
Khi đã thiết lập liên kết ở lần đầu, từ các lần sau bạn chỉ cần gõ ngắn gọn:

**macOS & Windows**
```bash
git push
```

*(Command này giống nhau trên macOS và Windows)*

**Kết quả kỳ vọng:**
```text
Enumerating objects: 4, done.
Counting objects: 100% (4/4), done.
Writing objects: 100% (3/3), 290 bytes | 290.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
To https://github.com/<username>/du-an-dau-tien.git
   3b8c9d2..7f2e1a4  main -> main
```

## 2. Kiểm tra trên website GitHub
Mở lại trình duyệt web tại trang repository của bạn và bấm phím **F5** (Tải lại trang):
* Bạn sẽ thấy file `hello.txt` đã xuất hiện trên giao diện GitHub.
* Số lượng commit hiển thị đã tăng lên.

---

# PHẦN 13 — LẤY DỮ LIỆU MỚI NHẤT TỪ GITHUB VỀ (GIT PULL)

Khi bạn làm việc nhóm hoặc khi bạn tự sửa một file trực tiếp trên giao diện web của GitHub, mã nguồn trên GitHub sẽ mới hơn mã nguồn trên máy tính của bạn.

Lúc này, bạn dùng lệnh `git pull` để tải và gộp các thay đổi mới nhất về máy cá nhân:

```text
GitHub Repository (Đám mây) ────[ git pull ]────> Local Repository (Máy bạn)
```

## Câu lệnh thực hiện:

**macOS & Windows**
```bash
git pull
```

*(Command này giống nhau trên macOS và Windows)*

**Kết quả kỳ vọng:**
```text
Updating 7f2e1a4..9d4f2b1
Fast-forward
 README.md | 3 +++
 1 file changed, 3 insertions(+)
```

### So sánh chiều dữ liệu giữa Push và Pull:
* `git push`: Đẩy đi (Từ **Local** lên **GitHub**).
* `git pull`: Kéo về (Từ **GitHub** về **Local**).

---

# PHẦN 14 — PHÂN BIỆT RÕ RỆT GIT CLONE VS GIT PULL

Người mới bắt đầu thường hay nhầm lẫn giữa hai lệnh này vì chúng đều "lấy code từ GitHub về máy". Hãy ghi nhớ bảng so sánh sau:

```text
┌─────────────────────────────────────────────────────────────┐
│                    GIT CLONE vs GIT PULL                    │
├──────────────────────────────┬──────────────────────────────┤
│          GIT CLONE           │           GIT PULL           │
├──────────────────────────────┼──────────────────────────────┤
│ Dùng khi CHƯA CÓ dự án trên  │ Dùng khi ĐÃ CÓ sẵn thư mục   │
│ máy tính.                    │ dự án trên máy tính.         │
│                              │                              │
│ Tải về toàn bộ repository và │ Chỉ tải về những commit mới  │
│ tự tạo một thư mục mới.      │ mà trên máy tính còn thiếu.  │
│                              │                              │
│ Chỉ chạy DUY NHẤT 1 LẦN đầu  │ Chạy HÀNG NGÀY mỗi khi bắt   │
│ tiên khi bắt đầu làm dự án.  │ đầu ngồi vào bàn làm việc.   │
└──────────────────────────────┴──────────────────────────────┘
```

---

# PHẦN 15 — QUẢN LÝ LIÊN KẾT MÁY CHỦ TỪ XA (GIT REMOTE)

Lệnh `git remote` quản lý địa chỉ máy chủ từ xa mà dự án của bạn đang kết nối tới.

## 1. Kiểm tra địa chỉ remote hiện tại

**macOS & Windows**
```bash
git remote -v
```

*(Command này giống nhau trên macOS và Windows)*

**Kết quả hiển thị:**
```text
origin  https://github.com/<username>/du-an-dau-tien.git (fetch)
origin  https://github.com/<username>/du-an-dau-tien.git (push)
```

## 2. Giải thích chữ `origin`
* `origin` chỉ đơn giản là một **tên gọi đại diện (nickname mặc định)** mà Git tự động gán cho đường dẫn repository GitHub khi bạn chạy lệnh `git clone`.
* Thay vì mỗi lần push/pull bạn phải gõ lại cả đường link dài `https://github.com/...`, bạn chỉ cần gọi tên ngắn gọn là `origin`.

---

# PHẦN 16 — LÀM VIỆC VỚI NHÁNH CƠ BẢN (BRANCH & MERGE)

## 1. Nhánh (Branch) là gì? Tại sao cần dùng nhánh?
* Mặc định, dự án có một nhánh chính là `main` (chứa mã nguồn ổn định nhất để chạy thực tế).
* Khi cần làm một tính năng mới (ví dụ làm nút Đăng nhập), bạn không nên code trực tiếp lên `main` vì nếu lỗi sẽ làm hỏng toàn bộ dự án.
* Bạn sẽ **tách một nhánh riêng** (ví dụ `feature-login`), thỏa sức thử nghiệm trên đó. Khi tính năng hoàn tất và đã kiểm tra kỹ lưỡng, bạn sẽ **gộp (merge)** nhánh đó trở lại vào `main`.

```text
main:       ●───────●────────────────● (Merge nhánh tính năng vào)
                     \              /
feature:              ●────────────● (Phát triển tính năng mới độc lập)
```

---

## 2. Xem danh sách các nhánh hiện có

**macOS & Windows**
```bash
git branch
```
*Nhánh có dấu sao `*` màu xanh phía trước là nhánh bạn đang đứng.*

---

## 3. Tạo một nhánh mới và chuyển ngay sang nhánh đó
Sử dụng lệnh `git switch -c` (*chữ `c` viết tắt của create - tạo mới*):

**macOS & Windows**
```bash
git switch -c <tên-nhánh-mới>
```

**Ví dụ:**
```bash
git switch -c feature-tinh-nang-moi
```

**Kết quả hiển thị:**
```text
Switched to a new branch 'feature-tinh-nang-moi'
```

---

## 4. Chuyển đổi qua lại giữa các nhánh
Khi muốn quay trở về lại nhánh chính `main`:

**macOS & Windows**
```bash
git switch main
```

---

## 5. Gộp nhánh (Merge)
Muốn gộp mã nguồn từ nhánh `feature-tinh-nang-moi` vào nhánh `main`:
1. Chuyển về nhánh đích muốn nhận code (ở đây là `main`):
   ```bash
   git switch main
   ```
2. Thực hiện gộp nhánh:
   ```bash
   git merge feature-tinh-nang-moi
   ```

> [!NOTE]
> Các kỹ thuật phân nhánh nâng cao (Git Flow, Rebase, Resolve Conflicts chuyên sâu) là chủ đề nâng cao mà chúng ta sẽ tìm hiểu sau khi đã làm chủ quy trình cơ bản này.

---

# PHẦN 17 — SO SÁNH CHI TIẾT THAY ĐỔI (GIT DIFF)

Lệnh `git diff` giúp bạn nhìn thấy chi tiết từng dòng code vừa được thêm vào hoặc xóa đi trước khi quyết định lưu lại.

## 1. Xem các thay đổi chưa được đưa vào Staging Area (Chưa `git add`)

**macOS & Windows**
```bash
git diff
```

**Cách đọc kết quả:**
* Dòng có dấu trừ `-` và màu đỏ: Dòng vừa bị xóa đi.
* Dòng có dấu cộng `+` và màu xanh lá cây: Dòng mới được thêm vào.

---

## 2. Xem các thay đổi đã đưa vào Staging Area (Đã `git add` nhưng chưa `commit`)

**macOS & Windows**
```bash
git diff --staged
```

Lệnh này giúp bạn kiểm tra lại một lượt cuối cùng xem "giỏ hàng" của mình đã chuẩn bị đầy đủ và sạch sẽ chưa trước khi tiến hành thanh toán (`commit`).

---

# PHẦN 18 — LOẠI TRỪ FILE RÁC VỚI .GITIGNORE

Trong quá trình lập trình, dự án sẽ sinh ra rất nhiều file không nên và không bao giờ được đưa lên GitHub, ví dụ:
* Thư mục thư viện cài thêm (nặng hàng trăm MB, có thể tải lại bất cứ lúc nào).
* File cấu hình cá nhân của IDE/Editor.
* File chứa mật khẩu, mã khóa bảo mật bí mật (`.env`).
* File chạy thực thi sau khi biên dịch (`.exe`, `.o`, `.class`).

File `.gitignore` là một file văn bản đặt ở thư mục gốc của dự án, chứa danh sách tên các file hoặc thư mục mà bạn muốn Git bỏ qua hoàn toàn.

## 1. Cú pháp cơ bản trong file `.gitignore`
* Dấu `#`: Viết ghi chú (comment).
* Tên thư mục kèm dấu gạch chéo `/`: Bỏ qua cả thư mục đó (ví dụ: `node_modules/`).
* Ký tự đại diện sao `*`: Đại diện cho phần mở rộng bất kỳ (ví dụ: `*.log` bỏ qua toàn bộ file log).

---

## 2. File mẫu .gitignore cho dự án Python
Tạo một file có tên chính xác là `.gitignore` và dán nội dung sau:

```gitignore
# Môi trường ảo Python
.venv/
venv/
env/

# File rác sinh ra trong quá trình chạy
__pycache__/
*.pyc

# File chứa biến môi trường / mật khẩu bí mật
.env

# Cấu hình phần mềm VS Code
.vscode/
```

---

## 3. File mẫu .gitignore cho dự án C / C++

```gitignore
# File đối tượng sau khi biên dịch
*.o
*.obj

# File chạy thực thi chương trình
*.exe
*.out

# Thư mục build dự án
build/
bin/

# File cấu hình IDE
.vscode/
.idea/
```

---

# PHẦN 19 — THỰC HÀNH WORKFLOW THỰC TẾ HOÀN CHỈNH TỪ A ĐẾN Z

Hãy cùng nhau thực hành từng bước một quy trình làm việc thực tế từ đầu tới cuối trên máy của bạn.

```text
┌───────────────────────────────────────────────────────────┐
│              SƠ ĐỒ QUY TRÌNH LÀM VIỆC CHUẨN               │
│                                                           │
│  1. git clone <url>   (Tải dự án về máy - chỉ làm 1 lần)  │
│          ↓                                                │
│  2. cd <folder>       (Đi vào thư mục dự án)              │
│          ↓                                                │
│  3. [Viết / Sửa Code] (Mở file lên và làm việc)           │
│          ↓                                                │
│  4. git status        (Kiểm tra trạng thái thay đổi)      │
│          ↓                                                │
│  5. git add .         (Đưa toàn bộ thay đổi vào giỏ hàng) │
│          ↓                                                │
│  6. git commit -m     (Đóng gói commit có mô tả rõ ràng)  │
│          ↓                                                │
│  7. git push          (Đẩy code lên đám mây GitHub)       │
│          ↓                                                │
│  8. git pull          (Lấy code mới nhất về khi cần)      │
└───────────────────────────────────────────────────────────┘
```

---

## BƯỚC 1: Tải repository về Desktop

**macOS — Terminal**
```bash
cd ~/Desktop
git clone <repository-url>
```

**Windows — PowerShell**
```powershell
cd ~/Desktop
git clone <repository-url>
```

---

## BƯỚC 2: Di chuyển vào thư mục dự án

**macOS — Terminal**
```bash
cd <repository-name>
```

**Windows — PowerShell**
```powershell
cd <repository-name>
```

---

## BƯỚC 3: Tạo một file mới hoặc chỉnh sửa file

Hãy tạo một file đơn giản có tên `trang-chu.html`.

**macOS — Terminal (Tạo nhanh bằng lệnh)**
```bash
echo "<h1>Xin chao day la du an Git dau tien</h1>" > trang-chu.html
```

**Windows — PowerShell (Tạo nhanh bằng lệnh)**
```powershell
Set-Content -Path trang-chu.html -Value "<h1>Xin chao day la du an Git dau tien</h1>"
```

*(Hoặc bạn có thể mở thư mục bằng VS Code và dùng chuột tạo file mới bình thường).*

---

## BƯỚC 4: Kiểm tra trạng thái

**macOS & Windows**
```bash
git status
```
*Bạn sẽ thấy `trang-chu.html` hiển thị màu đỏ dưới mục `Untracked files`.*

---

## BƯỚC 5: Đưa file vào Staging Area

**macOS & Windows**
```bash
git add .
```

Kiểm tra lại:
```bash
git status
```
*File `trang-chu.html` đã chuyển sang màu xanh lá cây.*

---

## BƯỚC 6: Tạo Commit lưu vết

**macOS & Windows**
```bash
git commit -m "Them trang chu giao dien don gian"
```

Kiểm tra lịch sử:
```bash
git log --oneline
```

---

## BƯỚC 7: Đẩy mã nguồn lên GitHub

**macOS & Windows**
```bash
git push
```
*(Nếu là lần đầu đẩy nhánh này, dùng: `git push -u origin main`)*

Mở trang GitHub trên trình duyệt để chiêm ngưỡng file vừa được đẩy lên thành công!

---

## BƯỚC 8: Thử nghiệm tạo tính năng mới bằng Branch

1. Tạo và nhảy sang nhánh mới:
   ```bash
   git switch -c cap-nhat-giao-dien
   ```

2. Sửa tiếp file hoặc tạo thêm nội dung. Ví dụ thêm một dòng văn bản:
   * **macOS:** `echo "<p>Noi dung bo sung</p>" >> trang-chu.html`
   * **Windows:** `Add-Content -Path trang-chu.html -Value "<p>Noi dung bo sung</p>"`

3. Đóng gói trên nhánh mới:
   ```bash
   git add .
   git commit -m "Bo sung the doan van cho trang chu"
   ```

4. Chuyển về nhánh chính `main`:
   ```bash
   git switch main
   ```
   *(Lúc này nếu mở file `trang-chu.html` ra, bạn sẽ thấy dòng vừa thêm không hề có ở đây, vì nó nằm ở nhánh tính năng riêng).*

5. Gộp thay đổi vào `main`:
   ```bash
   git merge cap-nhat-giao-dien
   ```

6. Đẩy kết quả cuối cùng lên GitHub:
   ```bash
   git push
   ```

---

# PHẦN 20 — BẢNG TRA CỨU CÁC LỆNH GIT CỐT LÕI (CHEAT SHEET)

| Lệnh Git | Chức năng chi tiết |
| :--- | :--- |
| `git --version` | Kiểm tra phiên bản Git hiện có trên máy tính |
| `git config --global user.name "..."` | Khai báo tên tác giả cho các commit |
| `git config --global user.email "..."` | Khai báo email tác giả cho các commit |
| `git init` | Khởi tạo một Local Repository mới từ thư mục hiện tại |
| `git clone <url>` | Sao chép toàn bộ dự án từ GitHub về máy tính |
| `git status` | Xem trạng thái các file (mới, đã sửa, đã staged) |
| `git add <file>` | Đưa một file cụ thể vào vùng chuẩn bị (Staging Area) |
| `git add .` | Đưa toàn bộ các thay đổi vào vùng chuẩn bị |
| `git commit -m "..."` | Đóng gói và lưu vết lịch sử kèm thông điệp mô tả |
| `git push` | Đẩy các commit từ máy tính lên máy chủ GitHub |
| `git pull` | Lấy các commit mới nhất từ GitHub cập nhật vào máy |
| `git log --oneline` | Xem lịch sử các mốc commit một cách thu gọn |
| `git diff` | Xem chi tiết từng dòng thay đổi chưa được add |
| `git diff --staged` | Xem chi tiết từng dòng thay đổi đã được add |
| `git remote -v` | Xem danh sách các liên kết máy chủ từ xa |
| `git branch` | Xem danh sách các nhánh hiện có trong dự án |
| `git switch -c <nhánh>` | Tạo một nhánh mới và chuyển ngay sang nhánh đó |
| `git switch <nhánh>` | Chuyển đổi qua lại giữa các nhánh |
| `git merge <nhánh>` | Gộp mã nguồn từ nhánh khác vào nhánh hiện tại |

---

# PHẦN 21 — XỬ LÝ SỰ CỐ & LỖI THƯỜNG GẶP (TROUBLESHOOTING)

Trong quá trình học và làm việc, chắc chắn bạn hoặc người học sẽ gặp phải một số thông báo lỗi dưới đây. Hãy bình tĩnh tra cứu cách giải quyết tương ứng:

---

## 1. Lỗi trên hệ điều hành Windows

### Lỗi: `git : The term 'git' is not recognized as the name of a cmdlet...`
* **Nguyên nhân:** Máy tính chưa được cài Git, hoặc vừa cài xong nhưng chưa khởi động lại PowerShell nên hệ thống chưa nhận diện biến môi trường `PATH`.
* **Cách khắc phục:**
  1. Tắt toàn bộ cửa sổ PowerShell và mở lại một cửa sổ mới.
  2. Nếu vẫn bị, mở PowerShell và cài đặt lại bằng lệnh:
     ```powershell
     winget install --id Git.Git -e --source winget
     ```

### Lỗi: `gh : The term 'gh' is not recognized...`
* **Nguyên nhân:** Chưa cài GitHub CLI hoặc chưa tắt đi mở lại PowerShell sau khi cài.
* **Cách khắc phục:**
  1. Tắt và mở lại PowerShell.
  2. Nếu chưa có, gõ lệnh:
     ```powershell
     winget install --id GitHub.cli
     ```

### Lỗi: `winget : The term 'winget' is not recognized...`
* **Nguyên nhân:** Phiên bản Windows cũ chưa được cập nhật gói App Installer của Microsoft Store.
* **Cách khắc phục:** Tải bộ cài đặt Git trực tiếp từ trang web chính thức: [https://git-scm.com/download/win](https://git-scm.com/download/win) và cài đặt thủ công.

---

## 2. Lỗi trên hệ điều hành macOS

### Lỗi: `zsh: command not found: git`
* **Nguyên nhân:** Máy macOS chưa cài Xcode Command Line Tools.
* **Cách khắc phục:** Chạy lệnh:
  ```bash
  xcode-select --install
  ```
  và bấm **Install** trên hộp thoại xuất hiện.

### Lỗi: `zsh: command not found: gh`
* **Nguyên nhân:** Chưa cài GitHub CLI.
* **Cách khắc phục:** Chạy lệnh thông qua Homebrew:
  ```bash
  brew install gh
  ```

---

## 3. Lỗi xác thực và kết nối GitHub (Chung cho cả hai hệ điều hành)

### Lỗi: `fatal: Authentication failed` hoặc `Permission denied (publickey)`
* **Nguyên nhân:** Máy tính chưa đăng nhập tài khoản GitHub hoặc phiên đăng nhập đã hết hạn.
* **Cách khắc phục:** Chạy lại lệnh xác thực GitHub CLI:
  ```bash
  gh auth login
  ```
  Chọn đăng nhập qua trình duyệt web như hướng dẫn ở Phần 5.

### Lỗi: `fatal: repository 'https://github.com/...' not found`
* **Nguyên nhân:**
  1. Bạn copy thiếu hoặc gõ sai địa chỉ URL của repository.
  2. Repository ở chế độ **Private** và bạn chưa đăng nhập tài khoản có quyền truy cập vào repo đó.
* **Cách khắc phục:**
  1. Kiểm tra lại thật kỹ đường link HTTPS trên trang GitHub.
  2. Chạy `gh auth status` để đảm bảo đang đăng nhập đúng tài khoản sở hữu repo.

---

## 4. Lỗi Push bị từ chối (Updates were rejected because the remote contains work that you do not have locally)
* **Hiện tượng:** Khi gõ `git push`, màn hình báo lỗi đỏ và từ chối không cho đẩy code lên.
* **Nguyên nhân:** Trên GitHub đang có những thay đổi mới (ví dụ ai đó vừa push lên, hoặc bạn vừa chỉnh sửa file README trên web) mà máy tính của bạn chưa tải về. Git ngăn chặn để tránh việc bạn vô tình ghi đè làm mất code của người khác.
* **Cách khắc phục:** Kéo các thay đổi mới trên GitHub về máy trước:
  ```bash
  git pull
  ```
  Sau khi Git đã gộp xong nội dung mới về máy, bạn thực hiện đẩy lại:
  ```bash
  git push
  ```

---

## 5. Lỗi xung đột mã nguồn (Merge Conflict) khi `git pull`
* **Hiện tượng:** Khi chạy `git pull`, Git báo `CONFLICT (content): Merge conflict in <tên-file>`.
* **Nguyên nhân:** Cả bạn và đồng đội (hoặc chính bạn trên web) cùng chỉnh sửa vào **cùng một dòng** trong cùng một file. Git không thể tự quyết định dòng nào đúng, dòng nào sai nên dừng lại để bạn tự lựa chọn.
* **Cách nhận biết trong file code:** Mở file bị conflict lên trong editor (VS Code), bạn sẽ thấy các ký hiệu lạ:
  ```text
  <<<<<<< HEAD
  Dòng code hiện tại trên máy của bạn
  =======
  Dòng code mới được kéo từ GitHub về
  >>>>>>> 9d4f2b1
  ```
* **Cách khắc phục:**
  1. Đọc kỹ hai đoạn code, xóa các ký hiệu `<<<<<<<`, `=======`, `>>>>>>>` và chỉ giữ lại đoạn code đúng nhất theo ý bạn.
  2. Lưu file lại.
  3. Đóng gói và hoàn tất việc sửa conflict:
     ```bash
     git add .
     git commit -m "Giai quyet conflict merge"
     git push
     ```

---

## 6. Các tính năng nâng cao (Để dành học sau khi đã thành thạo)

Khi mới bắt đầu tiếp cận Git & GitHub, bạn **KHÔNG CẦN** và **CHƯA NÊN** vội tìm hiểu các chủ đề sau để tránh bị rối và quá tải thông tin:

* **Tính năng Git nâng cao:** `git rebase`, `git cherry-pick`, `git bisect`, `git submodule`, `git worktree`, `git reflog`, Git hooks, cấu trúc lưu trữ nội bộ của Git (Git internals).
* **Tính năng GitHub mở rộng:** GitHub Actions (tự động hóa CI/CD), GitHub Issues (quản lý lỗi), GitHub Projects (quản lý công việc kiểu Kanban), GitHub Discussions, GitHub Pages (host web miễn phí), GitHub Packages, GitHub Codespaces.

> [!TIP]
> Tất cả các kỹ thuật trên đều hữu ích trong môi trường doanh nghiệp lớn, nhưng **chỉ nên học sau khi bạn đã làm chủ 100% workflow cơ bản** được hướng dẫn trong tài liệu này!

---

## TỔNG KẾT: TÂM THẾ KHI SỬ DỤNG GIT
Khi mới tiếp cận, bạn chỉ cần ghi nhớ đúng một chu trình 5 bước kinh điển lặp đi lặp lại mỗi ngày:

```text
git status   →  Kiểm tra những gì mình vừa sửa
git add .    →  Cho toàn bộ vào giỏ hàng
git commit   →  Đóng gói mốc lịch sử kèm lời nhắn
git push     →  Đẩy lên mây để lưu trữ và chia sẻ
git pull     →  Lấy phiên bản mới nhất về máy khi bắt đầu ngày làm việc
```

Chúc bạn và người học thực hành thuận lợi và tự tin làm chủ Git & GitHub trong mọi dự án thực tế!
