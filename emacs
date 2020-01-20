(custom-set-variables
  ;; custom-set-variables was added by Custom.
  ;; If you edit it by hand, you could mess it up, so be careful.
  ;; Your init file should contain only one such instance.
  ;; If there is more than one, they won't work right.
 '(inhibit-startup-screen t))
(custom-set-faces
  ;; custom-set-faces was added by Custom.
  ;; If you edit it by hand, you could mess it up, so be careful.
  ;; Your init file should contain only one such instance.
  ;; If there is more than one, they won't work right.
 )
;;; Matlab-mode setup:

;; Add local lisp folder to load-path
(setq load-path (append load-path (list "~/elisp")))

;; Set up matlab-mode to load on .m files
(autoload 'matlab-mode "matlab" "Enter MATLAB mode." t)
(setq auto-mode-alist (cons '("\\.m\\'" . matlab-mode) auto-mode-alist))
(autoload 'matlab-shell "matlab" "Interactive MATLAB mode." t)

; matlab settings
(autoload 'matlab-mode "/home/skhosh/elisp/matlab.el" "Enter Matlab mode." t)
(setq auto-mode-alist (cons '("\\.m\\'" . matlab-mode) auto-mode-alist))
(autoload 'matlab-shell "/home/skhosh/elisp/matlab.el" "Interactive Matlab mode." t)
(setq matlab-shell-command "/home/skhosh/elisp/mymatlab.sh") 
(setq matlab-shell-command-switches "")
;;;
;;;
(setq auto-mode-alist
      (append '(("\\.F90$" . f90-mode))    ;; xxx.F90: preprocessor -> compile
              auto-mode-alist))
;;;
(setq auto-mode-alist (cons '(".F90$" . f90-mode) auto-mode-alist))
(setq auto-mode-alist (cons '(".F90$" . f90-mode) auto-mode-alist))
(setq auto-mode-alist (cons '("\\.m$" . octave-mode) auto-mode-alist))
;;;
(set-face-attribute 'default (selected-frame) :height 100)
;;;
(set-face-attribute 'default nil :height 150)
;;;
(global-set-key (kbd "M-/") 'goto-line)
(put 'upcase-region 'disabled nil)
(put 'downcase-region 'disabled nil)
