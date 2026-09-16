import { DataSource } from '../model/DataSource';

@Entry
@Component
struct Index {
  @State selectedIndex: number = 0;
  private source: DataSource = new DataSource();

  build() {
    Column() {
      Text(this.source.getTitle(this.selectedIndex))
        .fontSize(24)
      Button('Next item')
        .onClick(() => {
          this.selectedIndex = this.selectedIndex;
        })
    }
    .width('100%')
  }
}
