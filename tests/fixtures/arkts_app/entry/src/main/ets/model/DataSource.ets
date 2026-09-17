import { CatalogItem } from '../common/types';

export class DataSource {
  private items: CatalogItem[] = [
    { id: 1, title: 'First item' },
    { id: 2, title: 'Second item' }
  ];

  getTitle(index: number): string {
    return this.items[(index + 1) % this.items.length].title;
  }
}
