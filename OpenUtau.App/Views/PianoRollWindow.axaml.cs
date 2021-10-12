﻿using System;
using System.ComponentModel;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Controls.Shapes;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;
using Avalonia.VisualTree;
using OpenUtau.App.Controls;
using OpenUtau.App.ViewModels;

namespace OpenUtau.App.Views {
    public partial class PianoRollWindow : Window {
        public MainWindow? MainWindow { get; set; }
        public readonly PianoRollViewModel ViewModel;

        private NoteEditState? editState;
        private Rectangle? selectionBox;
        private Border? expValueTip;
        private LyricBox? lyricBox;

        public PianoRollWindow() {
            InitializeComponent();
#if DEBUG
            this.AttachDevTools();
#endif
            DataContext = ViewModel = new PianoRollViewModel();
        }

        private void InitializeComponent() {
            AvaloniaXamlLoader.Load(this);
            expValueTip = this.FindControl<Border>("ExpValueTip");
            lyricBox = this.FindControl<LyricBox>("LyricBox");
        }

        public void WindowDeactivated(object sender, EventArgs args) {
            //lyricBox?.EndEdit();
        }

        void WindowClosing(object? sender, CancelEventArgs e) {
            Hide();
            e.Cancel = true;
        }

        void OnMenuRenamePart(object? sender, RoutedEventArgs e) {
            var part = ViewModel.NotesViewModel.Part;
            if (part == null) {
                return;
            }
            var dialog = new TypeInDialog();
            dialog.Title = "Rename";
            dialog.SetText(part.name);
            dialog.onFinish = name => {
                if (!string.IsNullOrWhiteSpace(name) && name != part.name) {
                    ViewModel.RenamePart(part, name);
                }
            };
            dialog.Show(this);
        }

        public void HScrollPointerWheelChanged(object sender, PointerWheelEventArgs args) {
            var scrollbar = (ScrollBar)sender;
            scrollbar.Value = Math.Max(scrollbar.Minimum, Math.Min(scrollbar.Maximum, scrollbar.Value - scrollbar.SmallChange * args.Delta.Y));
            lyricBox?.EndEdit();
        }

        public void VScrollPointerWheelChanged(object sender, PointerWheelEventArgs args) {
            var scrollbar = (ScrollBar)sender;
            scrollbar.Value = Math.Max(scrollbar.Minimum, Math.Min(scrollbar.Maximum, scrollbar.Value - scrollbar.SmallChange * args.Delta.Y));
            lyricBox?.EndEdit();
        }

        public void TimelinePointerWheelChanged(object sender, PointerWheelEventArgs args) {
            var canvas = (Canvas)sender;
            var position = args.GetCurrentPoint((IVisual)sender).Position;
            var size = canvas.Bounds.Size;
            position = position.WithX(position.X / size.Width).WithY(position.Y / size.Height);
            ViewModel.NotesViewModel.OnXZoomed(position, 0.1 * args.Delta.Y);
            lyricBox?.EndEdit();
        }

        public void ViewScalerPointerWheelChanged(object sender, PointerWheelEventArgs args) {
            ViewModel.NotesViewModel.OnYZoomed(new Point(0, 0.5), 0.1 * args.Delta.Y);
            lyricBox?.EndEdit();
        }

        public void TimelinePointerPressed(object sender, PointerPressedEventArgs args) {
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            if (point.Properties.IsLeftButtonPressed) {
                args.Pointer.Capture(canvas);
                int tick = ViewModel.NotesViewModel.PointToSnappedTick(point.Position);
                ViewModel.PlaybackViewModel?.MovePlayPos(tick);
            }
            lyricBox?.EndEdit();
        }

        public void TimelinePointerMoved(object sender, PointerEventArgs args) {
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            if (point.Properties.IsLeftButtonPressed) {
                int tick = ViewModel.NotesViewModel.PointToSnappedTick(point.Position);
                ViewModel.PlaybackViewModel?.MovePlayPos(tick);
            }
        }

        public void TimelinePointerReleased(object sender, PointerReleasedEventArgs args) {
            args.Pointer.Capture(null);
        }

        public void NotesCanvasPointerPressed(object sender, PointerPressedEventArgs args) {
            lyricBox?.EndEdit();
            if (ViewModel.NotesViewModel.Part == null) {
                return;
            }
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            if (editState != null) {
                return;
            }
            if (point.Properties.IsLeftButtonPressed) {
                NotesCanvasLeftPointerPressed(canvas, point, args);
            } else if (point.Properties.IsRightButtonPressed) {
                ViewModel.NotesViewModel.DeselectNotes();
                editState = new NoteEraseEditState(canvas, ViewModel);
                Cursor = ViewConstants.cursorNo;
            } else if (point.Properties.IsMiddleButtonPressed) {
                editState = new NotePanningState(canvas, ViewModel);
                Cursor = ViewConstants.cursorHand;
            }
            if (editState != null) {
                editState.Begin(point.Pointer, point.Position);
                editState.Update(point.Pointer, point.Position);
            }
        }

        private void NotesCanvasLeftPointerPressed(Canvas canvas, PointerPoint point, PointerPressedEventArgs args) {
            var pitHitInfo = ViewModel.NotesViewModel.HitTest.HitTestPitchPoint(point.Position);
            if (pitHitInfo.Note != null) {
                editState = new PitchPointEditState(canvas, ViewModel,
                    pitHitInfo.Note, pitHitInfo.Index, pitHitInfo.OnPoint, pitHitInfo.X, pitHitInfo.Y);
                return;
            }
            var vbrHitInfo = ViewModel.NotesViewModel.HitTest.HitTestVibrato(point.Position);
            if (vbrHitInfo.hit) {
                if (vbrHitInfo.hitToggle) {
                    ViewModel.NotesViewModel.ToggleVibrato(vbrHitInfo.note);
                    return;
                }
                if (vbrHitInfo.hitStart) {
                    editState = new VibratoChangeStartState(canvas, ViewModel, vbrHitInfo.note);
                    return;
                }
                if (vbrHitInfo.hitIn) {
                    editState = new VibratoChangeInState(canvas, ViewModel, vbrHitInfo.note);
                    return;
                }
                if (vbrHitInfo.hitOut) {
                    editState = new VibratoChangeOutState(canvas, ViewModel, vbrHitInfo.note);
                    return;
                }
                if (vbrHitInfo.hitDepth) {
                    editState = new VibratoChangeDepthState(canvas, ViewModel, vbrHitInfo.note);
                    return;
                }
                if (vbrHitInfo.hitPeriod) {
                    editState = new VibratoChangePeriodState(canvas, ViewModel, vbrHitInfo.note);
                    return;
                }
                if (vbrHitInfo.hitShift) {
                    editState = new VibratoChangeShiftState(
                        canvas, ViewModel, vbrHitInfo.note, vbrHitInfo.point, vbrHitInfo.initialShift);
                    return;
                }
                return;
            }
            var noteHitInfo = ViewModel.NotesViewModel.HitTest.HitTestNote(point.Position);
            if (args.KeyModifiers == KeyModifiers.Control) {
                // New selection.
                ViewModel.NotesViewModel.DeselectNotes();
                editState = new NoteSelectionEditState(canvas, ViewModel, GetSelectionBox(canvas));
                Cursor = ViewConstants.cursorCross;
                return;
            }
            if (args.KeyModifiers == (KeyModifiers.Control | KeyModifiers.Shift)) {
                // Additional selection.
                editState = new NoteSelectionEditState(canvas, ViewModel, GetSelectionBox(canvas));
                Cursor = ViewConstants.cursorCross;
                return;
            }
            if (noteHitInfo.hitBody) {
                if (noteHitInfo.hitResizeArea) {
                    editState = new NoteResizeEditState(canvas, ViewModel, noteHitInfo.note);
                    Cursor = ViewConstants.cursorSizeWE;
                } else {
                    editState = new NoteMoveEditState(canvas, ViewModel, noteHitInfo.note);
                    Cursor = ViewConstants.cursorSizeAll;
                }
                return;
            }
            ViewModel.NotesViewModel.DeselectNotes();
            var note = ViewModel.NotesViewModel.MaybeAddNote(point.Position);
            if (note != null) {
                // Start moving right away
                editState = new NoteMoveEditState(canvas, ViewModel, note);
                Cursor = ViewConstants.cursorSizeAll;
            }
        }

        private Rectangle GetSelectionBox(Canvas canvas) {
            if (selectionBox != null) {
                return selectionBox;
            }
            selectionBox = new Rectangle() {
                Stroke = ThemeManager.ForegroundBrush,
                StrokeThickness = 2,
                Fill = ThemeManager.TickLineBrushLow,
                // radius = 8
                IsHitTestVisible = false,
            };
            canvas.Children.Add(selectionBox);
            selectionBox.ZIndex = 1000;
            return selectionBox;
        }

        public void NotesCanvasPointerMoved(object sender, PointerEventArgs args) {
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            if (editState != null) {
                editState.Update(point.Pointer, point.Position);
                return;
            }
            if (ViewModel?.NotesViewModel?.HitTest == null) {
                return;
            }
            var pitHitInfo = ViewModel.NotesViewModel.HitTest.HitTestPitchPoint(point.Position);
            if (pitHitInfo.Note != null) {
                Cursor = ViewConstants.cursorHand;
                return;
            }
            var vbrHitInfo = ViewModel.NotesViewModel.HitTest.HitTestVibrato(point.Position);
            if (vbrHitInfo.hit) {
                if (vbrHitInfo.hitDepth) {
                    Cursor = ViewConstants.cursorSizeNS;
                } else if (vbrHitInfo.hitPeriod) {
                    Cursor = ViewConstants.cursorSizeWE;
                } else {
                    Cursor = ViewConstants.cursorHand;
                }
                return;
            }
            var noteHitInfo = ViewModel.NotesViewModel.HitTest.HitTestNote(point.Position);
            if (noteHitInfo.hitResizeArea) {
                Cursor = ViewConstants.cursorSizeWE;
                return;
            }
            Cursor = null;
        }

        public void NotesCanvasPointerReleased(object sender, PointerReleasedEventArgs args) {
            if (editState == null) {
                return;
            }
            if (editState.MouseButton != args.InitialPressMouseButton) {
                return;
            }
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            editState.Update(point.Pointer, point.Position);
            editState.End(point.Pointer, point.Position);
            editState = null;
            Cursor = null;
        }

        public void NotesCanvasDoubleTapped(object sender, RoutedEventArgs args) {
            if (!(sender is Canvas canvas)) {
                return;
            }
            var e = (TappedEventArgs)args;
            var point = e.GetPosition(canvas);
            if (editState != null) {
                editState.End(e.Pointer, point);
                editState = null;
                Cursor = null;
            }
            var noteHitInfo = ViewModel.NotesViewModel.HitTest.HitTestNote(point);
            if (noteHitInfo.hitBody && ViewModel?.NotesViewModel?.Part != null) {
                var note = noteHitInfo.note;
                lyricBox?.Show(ViewModel.NotesViewModel.Part, note, note.lyric);
            }
        }

        public void NotesCanvasPointerWheelChanged(object sender, PointerWheelEventArgs args) {
            lyricBox?.EndEdit();
            if (args.KeyModifiers == KeyModifiers.None) {
                var scrollbar = this.FindControl<ScrollBar>("VScrollBar");
                VScrollPointerWheelChanged(scrollbar, args);
            } else if (args.KeyModifiers == KeyModifiers.Control) {
                var canvas = (Canvas)sender;
                var position = args.GetCurrentPoint((IVisual)sender).Position;
                var size = canvas.Bounds.Size;
                position = position.WithX(position.X / size.Width).WithY(position.Y / size.Height);
                ViewModel.NotesViewModel.OnYZoomed(position, 0.1 * args.Delta.Y);
            } else if (args.KeyModifiers == KeyModifiers.Shift) {
                var scrollbar = this.FindControl<ScrollBar>("HScrollBar");
                HScrollPointerWheelChanged(scrollbar, args);
            } else if (args.KeyModifiers == (KeyModifiers.Shift | KeyModifiers.Control)) {
                var canvas = this.FindControl<Canvas>("TimelineCanvas");
                TimelinePointerWheelChanged(canvas, args);
            }
        }

        public void ExpCanvasPointerPressed(object sender, PointerPressedEventArgs args) {
            lyricBox?.EndEdit();
            if (ViewModel.NotesViewModel.Part == null) {
                return;
            }
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            if (editState != null) {
                return;
            }
            if (point.Properties.IsLeftButtonPressed) {
                editState = new ExpSetValueState(canvas, ViewModel, expValueTip!);
            } else if (point.Properties.IsRightButtonPressed) {
                editState = new ExpResetValueState(canvas, ViewModel);
                Cursor = ViewConstants.cursorNo;
            }
            if (editState != null) {
                editState.Begin(point.Pointer, point.Position);
                editState.Update(point.Pointer, point.Position);
            }
        }

        public void ExpCanvasPointerMoved(object sender, PointerEventArgs args) {
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            if (editState != null) {
                editState.Update(point.Pointer, point.Position);
            } else {
                Cursor = null;
            }
        }

        public void ExpCanvasPointerReleased(object sender, PointerReleasedEventArgs args) {
            if (editState == null) {
                return;
            }
            if (editState.MouseButton != args.InitialPressMouseButton) {
                return;
            }
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            editState.Update(point.Pointer, point.Position);
            editState.End(point.Pointer, point.Position);
            editState = null;
            Cursor = null;
        }

        public void PhonemeCanvasPointerPressed(object sender, PointerPressedEventArgs args) {
            lyricBox?.EndEdit();
            if (ViewModel?.NotesViewModel?.Part == null) {
                return;
            }
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            if (editState != null) {
                return;
            }
            if (point.Properties.IsLeftButtonPressed) {
                var hitInfo = ViewModel.NotesViewModel.HitTest.HitTestPhoneme(point.Position);
                if (hitInfo.hit) {
                    var phoneme = hitInfo.phoneme;
                    var note = phoneme.Parent;
                    var index = note.PhonemeOffset + note.phonemes.IndexOf(phoneme);
                    if (hitInfo.hitPosition) {
                        editState = new PhonemeMoveState(
                            canvas, ViewModel, note.Extends ?? note, index);
                    } else if (hitInfo.hitPreutter) {
                        editState = new PhonemeChangePreutterState(
                            canvas, ViewModel, note.Extends ?? note, phoneme, index);
                    } else if (hitInfo.hitOverlap) {
                        editState = new PhonemeChangeOverlapState(
                            canvas, ViewModel, note.Extends ?? note, phoneme, index);
                    }
                }
            } else if (point.Properties.IsRightButtonPressed) {
                editState = new PhonemeResetState(canvas, ViewModel);
                Cursor = ViewConstants.cursorNo;
            }
            if (editState != null) {
                editState.Begin(point.Pointer, point.Position);
                editState.Update(point.Pointer, point.Position);
            }
        }

        public void PhonemeCanvasPointerMoved(object sender, PointerEventArgs args) {
            if (ViewModel?.NotesViewModel?.Part == null) {
                return;
            }
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            if (editState != null) {
                editState.Update(point.Pointer, point.Position);
                return;
            }
            var hitInfo = ViewModel.NotesViewModel.HitTest.HitTestPhoneme(point.Position);
            if (hitInfo.hit) {
                Cursor = ViewConstants.cursorSizeWE;
            } else {
                Cursor = null;
            }
        }

        public void PhonemeCanvasPointerReleased(object sender, PointerReleasedEventArgs args) {
            if (editState == null) {
                return;
            }
            if (editState.MouseButton != args.InitialPressMouseButton) {
                return;
            }
            var canvas = (Canvas)sender;
            var point = args.GetCurrentPoint(canvas);
            editState.Update(point.Pointer, point.Position);
            editState.End(point.Pointer, point.Position);
            editState = null;
            Cursor = null;
        }

        void OnKeyDown(object sender, KeyEventArgs args) {
            lyricBox?.EndEdit();
            var notesVm = ViewModel.NotesViewModel;
            if (notesVm.Part == null) {
                return;
            }
            if (args.KeyModifiers == KeyModifiers.None) {
                switch (args.Key) {
                    case Key.Delete: notesVm.DeleteSelectedNotes(); break;
                    case Key.U: notesVm.ShowVibrato = !notesVm.ShowVibrato; break;
                    case Key.I: notesVm.ShowPitch = !notesVm.ShowPitch; break;
                    case Key.O: notesVm.ShowPhoneme = !notesVm.ShowPhoneme; break;
                    case Key.P: notesVm.IsSnapOn = !notesVm.IsSnapOn; break;
                    case Key.T: notesVm.ShowTips = !notesVm.ShowTips; break;
                    case Key.Up: notesVm.TransposeSelection(1); break;
                    case Key.Down: notesVm.TransposeSelection(-1); break;
                    case Key.Space:
                        if (ViewModel.PlaybackViewModel != null &&
                            !ViewModel.PlaybackViewModel.PlayOrPause()) {
                            MessageBox.Show(
                               this,
                               "dialogs.noresampler.message",
                               "dialogs.noresampler.caption",
                               MessageBox.MessageBoxButtons.Ok);
                        }
                        break;
                    default: break;
                }
            } else if (args.KeyModifiers == KeyModifiers.Control) {
                switch (args.Key) {
                    case Key.A: notesVm.SelectAllNotes(); break;
                    case Key.S: _ = MainWindow?.Save(); break;
                    case Key.Z: ViewModel.Undo(); break;
                    case Key.Y: ViewModel.Redo(); break;
                    case Key.C: notesVm.CopyNotes(); break;
                    case Key.X: notesVm.CutNotes(); break;
                    case Key.V: notesVm.PasteNotes(); break;
                    case Key.Up: notesVm.TransposeSelection(12); break;
                    case Key.Down: notesVm.TransposeSelection(-12); break;
                    default: break;
                }
            } else if (args.KeyModifiers == KeyModifiers.Alt) {
                switch (args.Key) {
                    case Key.F4:
                        Hide();
                        break;
                    default:
                        break;
                }
            }
        }
    }
}